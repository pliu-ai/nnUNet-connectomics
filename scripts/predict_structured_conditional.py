#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import multiprocessing
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import isdir, join, maybe_mkdir_p, save_json
from torch import nn
from tqdm import tqdm

from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor, _parse_border_mirror_pad_size_arg
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.file_path_utilities import get_output_folder
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import LabelManager


def _build_structured_11_label_manager() -> LabelManager:
    """Build a fixed 11-class label manager for structured conditional output."""
    labels: Dict[str, int] = {
        "background": 0,
        "ecs": 1,
        "pm": 2,
        "cyto": 3,
        "er_mem": 4,
        "er_lum": 5,
        "nucpl": 6,
        "cond_slot_1": 7,
        "cond_slot_2": 8,
        "cond_slot_3_or_mito_ribo": 9,
        "other": 10,
    }
    return LabelManager(label_dict=labels, regions_class_order=None, force_use_labels=True)


def _build_original_32_identity_label_manager() -> LabelManager:
    """
    Build a 32-class label manager (0..31) with identity inference nonlinearity.

    This is used to export merged all-group score maps as original CellMap labels.
    """
    labels: Dict[str, int] = {"background": 0}
    for label in range(1, 32):
        labels[f"label_{label}"] = label
    return LabelManager(
        label_dict=labels,
        regions_class_order=None,
        force_use_labels=True,
        inference_nonlin=lambda x: x,
    )


def _is_structured_conditional_trainer_name(trainer_name: str) -> bool:
    trainer_name = str(trainer_name)
    if "StructuredConditional" in trainer_name:
        return True
    # Explicit aliases that inherit structured-conditional no-slot3 behavior.
    return trainer_name in {
        "nnUNetTrainerMemLumConsistency",
        "nnUNetTrainerStructuredConditionalNoSlot3MemLumConsistency",
    }


def _is_no_slot3_structured_trainer_name(trainer_name: str) -> bool:
    trainer_name = str(trainer_name)
    if "StructuredConditionalNoSlot3" in trainer_name:
        return True
    # Explicit aliases that are implemented on top of no-slot3 trainer.
    return trainer_name in {
        "nnUNetTrainerMemLumConsistency",
        "nnUNetTrainerStructuredConditionalNoSlot3MemLumConsistency",
    }


def _pick_structured_to_original_mapper(trainer_name: str) -> Callable[[torch.Tensor, int, int], torch.Tensor]:
    """
    Select mapping function by trainer variant.

    - nnUNetTrainerStructuredConditional -> slot3 variant
    - nnUNetTrainerStructuredConditionalNoSlot3 -> no-slot3 variant
    """
    trainer_name = str(trainer_name)
    if _is_no_slot3_structured_trainer_name(trainer_name):
        from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.label_mapping_no_slot3 import (
            structured_prediction_to_original_labels,
        )

        return structured_prediction_to_original_labels

    from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.label_mapping import (
        structured_prediction_to_original_labels,
    )

    return structured_prediction_to_original_labels


def _pick_all_groups_reconstructor(
    trainer_name: str,
) -> Callable[[Dict[int, torch.Tensor], str], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Select all-group reconstruction function by trainer variant.
    """
    trainer_name = str(trainer_name)
    if _is_no_slot3_structured_trainer_name(trainer_name):
        from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.inference_structured_conditional_no_slot3 import (
            reconstruct_original_labels_from_all_groups,
        )

        return reconstruct_original_labels_from_all_groups

    from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.inference_structured_conditional import (
        reconstruct_original_labels_from_all_groups,
    )

    return reconstruct_original_labels_from_all_groups


class StructuredConditionalPredictor(nnUNetPredictor):
    """
    Dedicated predictor for structured conditional trainers.

    This class does not change the generic nnUNet predictor behavior globally.
    It is only used by this script.
    """

    def __init__(
        self,
        group_id: int,
        run_all_groups: bool,
        output_mode: str,
        fixed_merge_mode: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.group_id = int(group_id)
        self.run_all_groups = bool(run_all_groups)
        self.output_mode = str(output_mode)
        self.fixed_merge_mode = str(fixed_merge_mode).lower().strip()
        if self.fixed_merge_mode not in ("max", "mean"):
            raise ValueError(
                f"fixed_merge_mode must be 'max' or 'mean', got {self.fixed_merge_mode!r}"
            )
        self.num_dynamic_groups = 12
        self.structured_label_manager = _build_structured_11_label_manager()
        self.original_32_identity_label_manager = _build_original_32_identity_label_manager()
        self._structured_to_original_mapper: Optional[Callable[[torch.Tensor, int, int], torch.Tensor]] = None
        self._all_groups_reconstructor: Optional[
            Callable[[Dict[int, torch.Tensor], str], Tuple[torch.Tensor, torch.Tensor]]
        ] = None

    def initialize_from_trained_model_folder(
        self,
        model_training_output_dir: str,
        use_folds: Union[Tuple[Union[int, str]], None],
        checkpoint_name: str = "checkpoint_final.pth",
    ):
        super().initialize_from_trained_model_folder(model_training_output_dir, use_folds, checkpoint_name)
        self._structured_to_original_mapper = _pick_structured_to_original_mapper(self.trainer_name)
        self._all_groups_reconstructor = _pick_all_groups_reconstructor(self.trainer_name)

        if not _is_structured_conditional_trainer_name(self.trainer_name):
            raise RuntimeError(
                f"This dedicated predictor expects a structured conditional trainer, got trainer_name={self.trainer_name}."
            )

    def _make_group_ids(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.full((int(batch_size),), int(self.group_id), dtype=torch.long, device=device)

    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        """Run conditioned forward pass with optional mirror TTA."""
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        group_ids = self._make_group_ids(x.shape[0], x.device)

        prediction = self.network(x, group_ids)
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]

        if mirror_axes is not None:
            assert max(mirror_axes) <= x.ndim - 3, "mirror_axes does not match the input dimensionality"

            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for axes in axes_combinations:
                pred_mirror = self.network(torch.flip(x, axes), group_ids)
                if isinstance(pred_mirror, (list, tuple)):
                    pred_mirror = pred_mirror[0]
                prediction += torch.flip(pred_mirror, axes)
            prediction /= float(len(axes_combinations) + 1)

        return prediction

    def _internal_predict_sliding_window_return_logits(
        self,
        data: torch.Tensor,
        slicers,
        do_on_device: bool = True,
    ):
        """
        Same logic as nnUNetPredictor, but channel allocation is derived from the
        model output (first tile) rather than label_manager.num_segmentation_heads.

        This avoids channel mismatch for fixed-head structured models.
        """
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device("cpu")

        try:
            empty_cache(self.device)

            if self.verbose:
                print(f"move image to device {results_device}")
            data = data.to(results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(
                    tuple(self.configuration_manager.patch_size),
                    sigma_scale=1.0 / 8.0,
                    value_scaling_factor=10,
                    device=results_device,
                )
            else:
                gaussian = 1

            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            if not self.allow_tqdm and self.verbose:
                print(f"running prediction: {len(slicers)} steps")

            for sl in tqdm(slicers, disable=not self.allow_tqdm):
                workon = data[sl][None].to(self.device)
                prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)

                if self.use_gaussian:
                    prediction *= gaussian

                # Lazy channel allocation based on first tile prediction.
                if predicted_logits is None:
                    predicted_logits = torch.zeros(
                        (int(prediction.shape[0]), *data.shape[1:]),
                        dtype=torch.half,
                        device=results_device,
                    )

                predicted_logits[sl] += prediction
                n_predictions[sl[1:]] += gaussian

            if predicted_logits is None:
                raise RuntimeError("No prediction tiles were generated; check input shape and patch size.")

            predicted_logits /= n_predictions
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError(
                    "Encountered inf in predicted logits. Reduce gaussian scaling or use higher-precision accumulator."
                )
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e

        return predicted_logits

    def _decode_one_group_logits_to_output_labels(
        self,
        logits: torch.Tensor,
        props: dict,
        group_id: int,
        num_processes_segmentation_export: int,
    ) -> np.ndarray:
        """
        Decode one group's 11-channel logits into final output labels.

        output_mode:
        - structured: returns structured labels [0..10]
        - original: returns mapped original CellMap labels for the selected group
        """
        seg_struct = convert_predicted_logits_to_segmentation_with_correct_shape(
            logits,
            self.plans_manager,
            self.configuration_manager,
            self.structured_label_manager,
            props,
            return_probabilities=False,
            num_threads_torch=max(1, int(num_processes_segmentation_export)),
        )
        if self.output_mode != "original":
            return np.asarray(seg_struct, dtype=np.uint8)

        seg_t = torch.from_numpy(np.asarray(seg_struct, dtype=np.int64))
        seg_out_t = self._structured_to_original_mapper(
            structured_prediction=seg_t,
            group_id=int(group_id),
            background_value=0,
        )
        return seg_out_t.cpu().numpy().astype(np.uint8)

    def _decode_all_groups_merged_original(
        self,
        logits_by_group: Dict[int, torch.Tensor],
        props: dict,
        num_processes_segmentation_export: int,
    ) -> np.ndarray:
        """
        Merge 12 conditioned runs and export one original-label segmentation.
        """
        _, score_map = self._all_groups_reconstructor(
            logits_by_group,
            fixed_merge_mode=self.fixed_merge_mode,
        )
        merged_scores = score_map[0]
        seg_merged = convert_predicted_logits_to_segmentation_with_correct_shape(
            merged_scores,
            self.plans_manager,
            self.configuration_manager,
            self.original_32_identity_label_manager,
            props,
            return_probabilities=False,
            num_threads_torch=max(1, int(num_processes_segmentation_export)),
        )
        return np.asarray(seg_merged, dtype=np.uint8)

    def predict_from_data_iterator(
        self,
        data_iterator,
        save_probabilities: bool = False,
        num_processes_segmentation_export: int = default_num_processes,
    ):
        """
        Dedicated export path that always decodes 11-channel structured logits first.

        output_mode:
        - structured: save structured labels [0..10]
        - original: map structured labels back to original CellMap IDs for selected group
        """
        if save_probabilities:
            raise NotImplementedError(
                "save_probabilities is not supported in this dedicated structured predictor."
            )

        if self._structured_to_original_mapper is None or self._all_groups_reconstructor is None:
            raise RuntimeError("Predictor not initialized. Call initialize_from_trained_model_folder first.")

        rw = self.plans_manager.image_reader_writer_class()
        output_file_ending = self.dataset_json["file_ending"]
        exported: List[np.ndarray] = []

        for preprocessed in data_iterator:
            data = preprocessed["data"]
            if isinstance(data, str):
                tmp_file = data
                data = torch.from_numpy(np.load(tmp_file))
                os.remove(tmp_file)

            ofile = preprocessed["ofile"]
            props = preprocessed["data_properties"]

            if ofile is not None:
                print(f"\nPredicting {os.path.basename(ofile)}")
            else:
                print(f"\nPredicting one in-memory case with shape={tuple(data.shape)}")

            if self.run_all_groups:
                logits_by_group: Dict[int, torch.Tensor] = {}
                parent_dir = os.path.dirname(ofile) if ofile is not None else None
                case_id = os.path.basename(ofile) if ofile is not None else None

                for group_id in range(self.num_dynamic_groups):
                    self.group_id = int(group_id)
                    logits = self.predict_logits_from_preprocessed_data(data).cpu()
                    logits_by_group[int(group_id)] = logits.unsqueeze(0)

                    seg_group = self._decode_one_group_logits_to_output_labels(
                        logits=logits,
                        props=props,
                        group_id=int(group_id),
                        num_processes_segmentation_export=num_processes_segmentation_export,
                    )

                    if ofile is not None:
                        group_dir = join(parent_dir, f"group_{group_id:02d}")
                        maybe_mkdir_p(group_dir)
                        rw.write_seg(seg_group, join(group_dir, case_id) + output_file_ending, props)

                seg_merged = self._decode_all_groups_merged_original(
                    logits_by_group=logits_by_group,
                    props=props,
                    num_processes_segmentation_export=num_processes_segmentation_export,
                )

                if ofile is not None:
                    rw.write_seg(seg_merged, ofile + output_file_ending, props)
                    print(f"done with {os.path.basename(ofile)} (group_00..group_11 + merged)")
                else:
                    exported.append(seg_merged)
            else:
                logits = self.predict_logits_from_preprocessed_data(data).cpu()
                seg_out = self._decode_one_group_logits_to_output_labels(
                    logits=logits,
                    props=props,
                    group_id=int(self.group_id),
                    num_processes_segmentation_export=num_processes_segmentation_export,
                )

                if ofile is not None:
                    rw.write_seg(seg_out, ofile + output_file_ending, props)
                    print(f"done with {os.path.basename(ofile)}")
                else:
                    exported.append(seg_out)

        return exported

    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) -> Union[np.ndarray, torch.Tensor]:
        """Keep base behavior but call the overridden conditioned sliding-window kernel."""
        with torch.no_grad():
            assert isinstance(input_image, torch.Tensor)
            self.network = self.network.to(self.device)
            self.network.eval()

            empty_cache(self.device)

            amp_context = (
                torch.autocast(self.device.type, enabled=True)
                if self.device.type == "cuda"
                else dummy_context()
            )
            with amp_context:
                assert input_image.ndim == 4, "input_image must be 4D tensor (c, x, y, z)"

                if self.verbose:
                    print(f"Input shape: {tuple(input_image.shape)}")
                    print(f"step_size: {self.tile_step_size}")
                    print(f"mirror_axes: {self.allowed_mirroring_axes if self.use_mirroring else None}")
                    print(f"padding_mode: {self.inference_padding_mode}")
                    print(f"border_mirror_pad_size: {self.inference_border_mirror_pad_size}")

                input_image_after_border_pad, slicer_revert_border_padding = \
                    self._maybe_apply_border_mirror_padding(input_image)
                data, slicer_revert_padding = self._pad_for_sliding_window(input_image_after_border_pad)
                slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

                if self.perform_everything_on_device and self.device != "cpu":
                    try:
                        predicted_logits = self._internal_predict_sliding_window_return_logits(
                            data,
                            slicers,
                            self.perform_everything_on_device,
                        )
                    except RuntimeError:
                        print("GPU result accumulation failed, falling back to CPU accumulation")
                        empty_cache(self.device)
                        predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, False)
                else:
                    predicted_logits = self._internal_predict_sliding_window_return_logits(
                        data,
                        slicers,
                        self.perform_everything_on_device,
                    )

                empty_cache(self.device)
                predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
                if slicer_revert_border_padding is not None:
                    predicted_logits = predicted_logits[(slice(None), *slicer_revert_border_padding[1:])]

        return predicted_logits


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dedicated inference for structured-conditional nnUNet trainers. "
            "This does not modify default nnUNetv2_predict behavior."
        )
    )
    parser.add_argument("-i", type=str, required=True, help="Input folder with *_0000 files.")
    parser.add_argument("-o", type=str, required=True, help="Output folder.")

    parser.add_argument(
        "-m",
        type=str,
        required=False,
        default=None,
        help="Trained model folder. If not set, -d/-tr/-p/-c are used to build the path.",
    )
    parser.add_argument("-d", type=str, required=False, default=None, help="Dataset id or dataset name.")
    parser.add_argument("-tr", type=str, required=False, default="nnUNetTrainerStructuredConditional")
    parser.add_argument("-p", type=str, required=False, default="nnUNetPlans")
    parser.add_argument("-c", type=str, required=False, default=None)

    parser.add_argument("-f", nargs="+", type=str, required=False, default=("all",))
    parser.add_argument("-chk", type=str, required=False, default="checkpoint_final.pth")
    parser.add_argument("-step_size", type=float, required=False, default=0.5)
    parser.add_argument("--disable_tta", action="store_true", required=False, default=False)
    parser.add_argument("--continue_prediction", action="store_true", required=False, default=False)
    parser.add_argument("--verbose", action="store_true", required=False, default=False)
    parser.add_argument("--disable_progress_bar", action="store_true", required=False, default=False)

    parser.add_argument("-npp", type=int, required=False, default=3)
    parser.add_argument("-nps", type=int, required=False, default=3)
    parser.add_argument("-num_parts", type=int, required=False, default=1)
    parser.add_argument("-part_id", type=int, required=False, default=0)
    parser.add_argument("-prev_stage_predictions", type=str, required=False, default=None)

    parser.add_argument("-device", type=str, default="cuda", required=False, choices=("cpu", "cuda", "mps"))
    parser.add_argument(
        "--inference_padding_mode",
        type=str,
        required=False,
        default="constant",
        choices=("constant", "reflect", "mirror"),
        help=(
            "Padding mode for sliding-window inference when input is smaller than patch size. "
            "Default: 'constant' (zero padding, original behavior). "
            "Set to 'reflect'/'mirror' for edge mirroring."
        ),
    )
    parser.add_argument(
        "--inference_border_mirror_pad_size",
        type=str,
        required=False,
        default="0",
        help=(
            "Optional border mirror padding size (voxels) applied to the whole input volume before "
            "sliding-window inference. Accepts one int (same for xyz) or 'x,y,z'. Default: 0"
        ),
    )

    parser.add_argument(
        "--group_id",
        type=str,
        required=False,
        default="0",
        help="Dynamic group ID (0..11), or 'all' to sweep all groups and write merged output.",
    )
    parser.add_argument(
        "--output_mode",
        type=str,
        required=False,
        default="original",
        choices=("structured", "original"),
        help="structured: save 0..10 structured labels; original: map back to original CellMap labels.",
    )
    parser.add_argument(
        "--fixed_merge_mode",
        type=str,
        required=False,
        default="mean",
        choices=("max", "mean"),
        help=(
            "How to merge fixed classes across group_00..group_11 when --group_id all. "
            "'mean': average confidence; 'max': maximum confidence."
        ),
    )
    return parser


def _resolve_model_folder(args: argparse.Namespace) -> str:
    if args.m is not None:
        return args.m
    if args.d is None or args.c is None:
        raise ValueError("Either provide -m, or provide both -d and -c (with optional -tr/-p).")
    return get_output_folder(args.d, args.tr, args.p, args.c)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        torch.set_num_threads(multiprocessing.cpu_count())
        return torch.device("cpu")
    if device_name == "cuda":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        return torch.device("cuda")
    return torch.device("mps")


def _parse_group_mode(group_id_arg: str) -> Tuple[int, bool]:
    """
    Parse --group_id into (group_id, run_all_groups).
    """
    value = str(group_id_arg).strip().lower()
    if value == "all":
        return 0, True

    gid = int(value)
    if gid < 0 or gid > 11:
        raise ValueError(f"--group_id must be in [0, 11] or 'all', got {group_id_arg!r}")
    return gid, False


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.part_id >= args.num_parts:
        raise ValueError("part_id must be < num_parts")

    args.f = [i if i == "all" else int(i) for i in args.f]
    model_folder = _resolve_model_folder(args)

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    device = _resolve_device(args.device)
    group_id, run_all_groups = _parse_group_mode(args.group_id)
    border_pad_size = _parse_border_mirror_pad_size_arg(args.inference_border_mirror_pad_size)

    predictor = StructuredConditionalPredictor(
        group_id=int(group_id),
        run_all_groups=bool(run_all_groups),
        output_mode=str(args.output_mode),
        fixed_merge_mode=str(args.fixed_merge_mode),
        tile_step_size=float(args.step_size),
        use_gaussian=True,
        use_mirroring=not args.disable_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=bool(args.verbose),
        verbose_preprocessing=bool(args.verbose),
        allow_tqdm=not args.disable_progress_bar,
        inference_padding_mode=str(args.inference_padding_mode),
        inference_border_mirror_pad_size=border_pad_size,
    )

    predictor.initialize_from_trained_model_folder(
        model_folder,
        args.f,
        checkpoint_name=str(args.chk),
    )

    # Save run metadata for reproducibility.
    save_json(
        {
            "model_folder": model_folder,
            "checkpoint": args.chk,
            "group_id": args.group_id,
            "run_all_groups": bool(run_all_groups),
            "output_mode": str(args.output_mode),
            "fixed_merge_mode": str(args.fixed_merge_mode),
            "inference_padding_mode": str(args.inference_padding_mode),
            "inference_border_mirror_pad_size": str(args.inference_border_mirror_pad_size),
            "trainer_name": str(predictor.trainer_name),
        },
        join(args.o, "structured_predict_config.json"),
        sort_keys=False,
    )

    print(
        "[StructuredConditionalPredict]"
        f" trainer={predictor.trainer_name}"
        f" group_id={args.group_id}"
        f" run_all_groups={run_all_groups}"
        f" output_mode={args.output_mode}"
        f" fixed_merge_mode={args.fixed_merge_mode}"
        f" inference_padding_mode={args.inference_padding_mode}"
        f" inference_border_mirror_pad_size={args.inference_border_mirror_pad_size}"
        f" checkpoint={args.chk}"
    )

    predictor.predict_from_files(
        args.i,
        args.o,
        save_probabilities=False,
        overwrite=not args.continue_prediction,
        num_processes_preprocessing=int(args.npp),
        num_processes_segmentation_export=int(args.nps),
        folder_with_segs_from_prev_stage=args.prev_stage_predictions,
        num_parts=int(args.num_parts),
        part_id=int(args.part_id),
    )


if __name__ == "__main__":
    main()
