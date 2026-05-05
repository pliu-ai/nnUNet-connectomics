from typing import Dict, List, Sequence, Tuple
import torch
from torch import autocast
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.training.loss.compound_losses import DC_CE_Partial_MergeProb_loss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.helpers import dummy_context
import numpy as np


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
        self.num_epochs = 1000
        self.began_save_chk = 800
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
