from typing import Dict, List, Sequence, Tuple
import torch
from torch import autocast
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.loss.compound_losses import DC_CE_Partial_MergeProb_loss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.helpers import dummy_context
import numpy as np


class nnUNetDataLoader3DPartialType(nnUNetDataLoader3D):
    """
    Draw one anchor case, then sample the remaining batch from the same partial-type bucket.
    This keeps most batches homogeneous in partial type and reduces repeated partial-loss calls.
    """
    def __init__(self,
                 data: nnUNetDataset,
                 batch_size: int,
                 patch_size,
                 final_patch_size,
                 label_manager,
                 case_to_partial_type: Dict[str, Tuple[int, ...]],
                 same_partial_type_batch_probability: float = 1.0,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities=None,
                 pad_sides=None,
                 transforms=None):
        super().__init__(data, batch_size, patch_size, final_patch_size, label_manager,
                         oversample_foreground_percent=oversample_foreground_percent,
                         sampling_probabilities=sampling_probabilities, pad_sides=pad_sides, transforms=transforms)
        self.case_to_partial_type = case_to_partial_type
        self.same_partial_type_batch_probability = float(np.clip(same_partial_type_batch_probability, 0.0, 1.0))
        self.partial_type_to_keys: Dict[Tuple[int, ...], List[str]] = {}
        for k in self.indices:
            p = self.case_to_partial_type.get(k)
            if p is None:
                continue
            p = tuple(p)
            self.partial_type_to_keys.setdefault(p, []).append(k)

        self._sampling_probability_by_key = None
        if self.sampling_probabilities is not None:
            self._sampling_probability_by_key = {
                k: float(v) for k, v in zip(self.indices, self.sampling_probabilities)
            }

    def _sample_random_indices(self) -> List[str]:
        return np.random.choice(self.indices, self.batch_size, replace=True, p=self.sampling_probabilities).tolist()

    def get_indices(self):
        # Keep base behavior for finite mode.
        if not self.infinite:
            return super().get_indices()

        # Fallback path: fully random batch.
        if len(self.partial_type_to_keys) == 0 or np.random.uniform() > self.same_partial_type_batch_probability:
            return self._sample_random_indices()

        anchor_key = self._sample_random_indices()[0]
        anchor_partial_type = self.case_to_partial_type.get(anchor_key)
        if anchor_partial_type is None:
            return self._sample_random_indices()

        candidate_keys = self.partial_type_to_keys.get(tuple(anchor_partial_type), [])
        if len(candidate_keys) == 0:
            return self._sample_random_indices()

        replace = len(candidate_keys) < self.batch_size
        if self._sampling_probability_by_key is None:
            return np.random.choice(candidate_keys, self.batch_size, replace=replace).tolist()

        candidate_probs = np.asarray(
            [self._sampling_probability_by_key.get(i, 0.0) for i in candidate_keys], dtype=np.float64
        )
        if candidate_probs.sum() <= 0:
            return np.random.choice(candidate_keys, self.batch_size, replace=replace).tolist()
        candidate_probs = candidate_probs / candidate_probs.sum()
        return np.random.choice(candidate_keys, self.batch_size, replace=replace, p=candidate_probs).tolist()


class nnUNetTrainerPartialLabel(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        #self.num_epochs = 250
        print("*"*10,"Using nnUNetTrainerPartialLabel","*"*10)
        self.case_to_partial_type_json = join(
            self.preprocessed_dataset_folder_base,
            "case_to_partial_type.json"
        )
        self.case_to_partial_type_dict = load_json(self.case_to_partial_type_json)
        self.num_iterations_per_epoch = 250 #250
        self.num_val_iterations_per_epoch = 10
        self.began_partial_epoch = 100
        self.num_epochs = 1500
        self.began_save_chk = 800
        self.same_partial_type_batch_probability = 1.0
        self.experiment_name = self.__class__.__name__ + "__" \
                            + configuration + "__" + f'fold_{fold}_MaxOnlyTumor'
        #if not continue_training:
       # self.wandb_logger = wandb.init(name=self.experiment_name,
       #                                     project="FLARE2023",
       #                                     config = self.configuration_manager)
       # 
        self.class_name = [key for key, value in sorted(self.dataset_json['labels'].items(), 
                                                        key=lambda item: item[1])]
        print("class name: ",self.class_name)
        self.ds_loss_weights = (1.0,)
        
        
    def _build_loss(self):
        loss = DC_CE_Partial_MergeProb_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, 
            {}, weight_ce=1, weight_dice=1,ignore_label=255, 
            dice_class=MemoryEfficientSoftDiceLoss)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))], dtype=np.float32)
            if self.is_ddp and not self._do_i_compile():
                # keep tiny non-zero tail to avoid occasional DDP unused parameter issues
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = tuple(float(i) for i in weights)
            print(f"deep supervision weights:{weights}")
        else:
            self.ds_loss_weights = (1.0,)

        return loss

    def _build_case_to_partial_type_lookup(self, dataset: nnUNetDataset) -> Dict[str, Tuple[int, ...]]:
        lookup = {}
        missing_keys = []
        for key in dataset.keys():
            found = None
            for candidate in self._normalize_case_key(key):
                if candidate in self.case_to_partial_type_dict:
                    found = self.case_to_partial_type_dict[candidate]
                    break
            if found is None:
                missing_keys.append(key)
                continue
            lookup[key] = tuple(self._parse_partial_type(found))

        if len(missing_keys) > 0:
            example = ", ".join(missing_keys[:5])
            raise KeyError(
                f"Missing partial type for {len(missing_keys)} training cases. Example keys: {example}"
            )
        return lookup

    def get_dataloaders(self):
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label
        )
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label
        )

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            dl_tr = nnUNetDataLoader2D(
                dataset_tr, self.batch_size, initial_patch_size, self.configuration_manager.patch_size,
                self.label_manager, oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None, pad_sides=None, transforms=tr_transforms
            )
            dl_val = nnUNetDataLoader2D(
                dataset_val, self.batch_size, self.configuration_manager.patch_size, self.configuration_manager.patch_size,
                self.label_manager, oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None, pad_sides=None, transforms=val_transforms
            )
        else:
            case_to_partial_type_lookup = self._build_case_to_partial_type_lookup(dataset_tr)
            dl_tr = nnUNetDataLoader3DPartialType(
                dataset_tr, self.batch_size, initial_patch_size, self.configuration_manager.patch_size,
                self.label_manager, case_to_partial_type=case_to_partial_type_lookup,
                same_partial_type_batch_probability=self.same_partial_type_batch_probability,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None, pad_sides=None, transforms=tr_transforms
            )
            dl_val = nnUNetDataLoader3D(
                dataset_val, self.batch_size, self.configuration_manager.patch_size, self.configuration_manager.patch_size,
                self.label_manager, oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None, pad_sides=None, transforms=val_transforms
            )

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr, transform=None, num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2), seeds=None,
                pin_memory=self.device.type == 'cuda', wait_time=0.002
            )
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val, transform=None, num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4), seeds=None,
                pin_memory=self.device.type == 'cuda', wait_time=0.002
            )

        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    @staticmethod
    def _normalize_case_key(case_key: str) -> List[str]:
        case_key = str(case_key)
        candidates = [case_key]
        for suffix in ('.nii.gz', '.nii', '.npz', '.npy', '.pkl'):
            if case_key.endswith(suffix):
                candidates.append(case_key[:-len(suffix)])
        if '.' in case_key:
            candidates.append(case_key.split('.')[0])
        # preserve order and remove duplicates
        seen = set()
        uniq = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq

    @staticmethod
    def _parse_partial_type(partial_type_value) -> List[int]:
        if isinstance(partial_type_value, str):
            raw_labels = [int(i) for i in partial_type_value.split('_') if i != ""]
        elif isinstance(partial_type_value, (int, np.integer)):
            raw_labels = [int(partial_type_value)]
        elif isinstance(partial_type_value, (list, tuple)):
            raw_labels = [int(i) for i in partial_type_value]
        else:
            raise TypeError(f"Unsupported partial type format: {type(partial_type_value)}")

        # partial type should contain foreground labels only
        raw_labels = [i for i in raw_labels if i != 0]
        seen = set()
        labels = []
        for i in raw_labels:
            if i not in seen:
                seen.add(i)
                labels.append(i)
        # Allow background-only samples (e.g. value '' or [0]). They are treated as
        # "no foreground labels are annotated" and will be grouped under empty tuple.
        return labels

    def _get_partial_types_for_keys(self, keys: Sequence[str]) -> List[List[int]]:
        partial_types = []
        for key in keys:
            found = None
            for candidate in self._normalize_case_key(key):
                if candidate in self.case_to_partial_type_dict:
                    found = self.case_to_partial_type_dict[candidate]
                    break
            if found is None:
                raise KeyError(
                    f"Cannot find partial type for case '{key}' in {self.case_to_partial_type_json}"
                )
            partial_types.append(self._parse_partial_type(found))
        return partial_types

    @staticmethod
    def _group_indices_by_partial_type(partial_types: Sequence[Sequence[int]]) -> Dict[Tuple[int, ...], List[int]]:
        groups: Dict[Tuple[int, ...], List[int]] = {}
        for sample_idx, partial_type in enumerate(partial_types):
            key = tuple(partial_type)
            if key not in groups:
                groups[key] = []
            groups[key].append(sample_idx)
        return groups

    def _compute_partial_loss(self, output, target, partial_types: Sequence[Sequence[int]]) -> torch.Tensor:
        outputs = list(output) if isinstance(output, (tuple, list)) else [output]
        targets = list(target) if isinstance(target, (tuple, list)) else [target]
        if len(outputs) != len(targets):
            raise RuntimeError(f"Output/target scale mismatch: {len(outputs)} vs {len(targets)}")

        if self.enable_deep_supervision and len(outputs) > 1:
            weights = self.ds_loss_weights
        else:
            weights = (1.0,)

        partial_groups = self._group_indices_by_partial_type(partial_types)
        total_loss = outputs[0].new_zeros(())

        for scale_idx, (scale_output, scale_target) in enumerate(zip(outputs, targets)):
            weight = weights[scale_idx] if scale_idx < len(weights) else 0.0
            if weight == 0:
                continue

            batch_size = int(scale_output.shape[0])
            scale_loss = scale_output.new_zeros(())
            for partial_type, sample_indices in partial_groups.items():
                if len(sample_indices) == 0:
                    continue
                idx = torch.as_tensor(sample_indices, device=scale_output.device, dtype=torch.long)
                grouped_output = scale_output.index_select(0, idx)
                grouped_target = scale_target.index_select(0, idx)
                group_weight = len(sample_indices) / batch_size
                scale_loss = scale_loss + group_weight * self.loss(grouped_output, grouped_target, list(partial_type))
            total_loss = total_loss + weight * scale_loss
        return total_loss
    
    def train_step(self, batch: dict, partial: bool=False) -> dict:
        data = batch['data']
        target = batch['target']
        keys = batch['keys']
        partial_types = self._get_partial_types_for_keys(keys)

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            l = self._compute_partial_loss(output, target, partial_types)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        keys = batch['keys']
        partial_types = self._get_partial_types_for_keys(keys)

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            l = self._compute_partial_loss(output, target, partial_types)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
