#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import List, Sequence, Tuple, Union

import torch
from batchgenerators.utilities.file_and_folder_operations import isdir, join, maybe_mkdir_p
from torch import nn
from torch._dynamo import OptimizedModule

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.file_path_utilities import get_output_folder


def _extract_condition_labels(dataset_json: dict) -> List[int]:
    labels = dataset_json.get("labels", {})
    vals: List[int] = []
    for name, value in labels.items():
        key = str(name).strip().lower()
        if key in {"background", "ignore"}:
            continue
        vals.append(int(value))
    vals = sorted(set(vals))
    if len(vals) == 0:
        raise RuntimeError("No foreground labels found in dataset_json")
    return vals


def _unwrap_for_methods(module: nn.Module) -> nn.Module:
    if isinstance(module, OptimizedModule):
        return module._orig_mod
    return module


def _try_set_condition_label_values(module: nn.Module, label_values: Sequence[int]) -> None:
    mod = _unwrap_for_methods(module)
    if hasattr(mod, "set_condition_label_values"):
        mod.set_condition_label_values(label_values)


class FixedConditionBinaryAsMulticlass(nn.Module):
    """
    Wrap a conditional binary network so predictor can still export with the standard
    multiclass label manager.

    For a fixed condition index:
    - run base network in binary mode (bg/fg)
    - compute fg score as (fg_logit - bg_logit)
    - place that score into the target class channel of a multiclass logits tensor
    - keep all other channels 0 (including background channel)
    """

    def __init__(self, base_network: nn.Module, condition_index: int, class_label: int, num_output_channels: int):
        super().__init__()
        self.base_network = base_network
        self.condition_index = int(condition_index)
        self.class_label = int(class_label)
        self.num_output_channels = int(num_output_channels)
        if not (0 <= self.class_label < self.num_output_channels):
            raise ValueError(
                f"class_label={self.class_label} out of range for num_output_channels={self.num_output_channels}"
            )

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.base_network.load_state_dict(state_dict, strict=strict)

    def state_dict(self, *args, **kwargs):
        return self.base_network.state_dict(*args, **kwargs)

    def _run_binary(self, x: torch.Tensor):
        cond = torch.full((x.shape[0],), self.condition_index, dtype=torch.long, device=x.device)
        base = _unwrap_for_methods(self.base_network)
        # ConditionalFiLMUNet signature: forward(x, condition)
        # DualDecoderConditionalFiLMUNet signature supports kwargs for binary output.
        try:
            out = base(x, condition=cond, return_binary=True, return_multiclass=False)
        except TypeError:
            out = base(x, cond)

        if isinstance(out, dict):
            if "binary" in out:
                out = out["binary"]
            else:
                raise RuntimeError("Unexpected dict output from conditional model (missing 'binary').")
        return out

    def _bin_to_multi(self, out_bin: torch.Tensor) -> torch.Tensor:
        if out_bin.shape[1] < 2:
            raise RuntimeError(f"Expected binary logits with >=2 channels, got shape={tuple(out_bin.shape)}")
        fg_score = out_bin[:, 1] - out_bin[:, 0]
        out = out_bin.new_zeros((out_bin.shape[0], self.num_output_channels, *out_bin.shape[2:]))
        out[:, self.class_label] = fg_score
        return out

    def forward(self, x: torch.Tensor):
        out_bin = self._run_binary(x)
        if isinstance(out_bin, list):
            return [self._bin_to_multi(o) for o in out_bin]
        return self._bin_to_multi(out_bin)


def _parse_labels_arg(labels: str | None) -> List[int] | None:
    if labels is None:
        return None
    parts = [p.strip() for p in labels.split(",") if p.strip()]
    if len(parts) == 0:
        return None
    return [int(p) for p in parts]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Per-condition native inference for conditional nnUNet models. "
            "Each condition is run through the model's binary head and exported as an individual mask folder."
        )
    )
    parser.add_argument("-i", type=str, required=True, help="Input folder with *_0000 files.")
    parser.add_argument("-o", type=str, required=True, help="Output root folder.")
    parser.add_argument(
        "--model_folder",
        type=str,
        default=None,
        help="Explicit trained model folder. If set, -d/-tr/-p/-c are ignored for model lookup.",
    )
    parser.add_argument("-d", type=str, required=True, help="Dataset id or name.")
    parser.add_argument("-p", type=str, required=False, default="nnUNetPlans", help="Plans identifier.")
    parser.add_argument("-tr", type=str, required=False, default="nnUNetTrainerConditionalFiLM", help="Trainer class.")
    parser.add_argument("-c", type=str, required=True, help="Configuration, e.g. 3d_lowres_large_patch.")
    parser.add_argument("-f", nargs="+", type=str, required=False, default=("all",), help="Folds, e.g. all or 0 1 2.")
    parser.add_argument("-chk", type=str, required=False, default="checkpoint_final.pth", help="Checkpoint filename.")
    parser.add_argument("-step_size", type=float, required=False, default=0.5, help="Sliding-window step size.")
    parser.add_argument("-npp", type=int, required=False, default=3, help="Preprocess workers.")
    parser.add_argument("-nps", type=int, required=False, default=3, help="Export workers.")
    parser.add_argument("--labels", type=str, default=None, help="Comma-separated label values to run. Default: all.")
    parser.add_argument("--disable_tta", action="store_true", default=False, help="Disable mirror TTA.")
    parser.add_argument("--continue_prediction", action="store_true", default=False, help="Skip existing outputs.")
    parser.add_argument("-num_parts", type=int, required=False, default=1, help="Parallel split count.")
    parser.add_argument("-part_id", type=int, required=False, default=0, help="Split id [0..num_parts-1].")
    parser.add_argument("-device", type=str, default="cuda", required=False, choices=("cpu", "cuda", "mps"))
    parser.add_argument("--disable_progress_bar", action="store_true", default=False)
    args = parser.parse_args()

    if args.part_id >= args.num_parts:
        raise ValueError("part_id must be < num_parts")

    args.f = [i if i == "all" else int(i) for i in args.f]

    if args.device == "cpu":
        import multiprocessing

        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device("cpu")
    elif args.device == "cuda":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device("cuda")
    else:
        device = torch.device("mps")

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    model_folder = args.model_folder if args.model_folder else get_output_folder(args.d, args.tr, args.p, args.c)
    predictor = nnUNetPredictor(
        tile_step_size=args.step_size,
        use_gaussian=True,
        use_mirroring=not args.disable_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=not args.disable_progress_bar,
    )
    predictor.initialize_from_trained_model_folder(model_folder, args.f, checkpoint_name=args.chk)

    condition_labels = _extract_condition_labels(predictor.dataset_json)
    selected_labels = _parse_labels_arg(args.labels)
    if selected_labels is None:
        selected_labels = condition_labels
    else:
        missing = [x for x in selected_labels if x not in condition_labels]
        if len(missing) > 0:
            raise ValueError(f"Requested labels not in condition label set: {missing}")

    # Make sure model label mapping is explicit and stable.
    _try_set_condition_label_values(predictor.network, condition_labels)

    print(
        f"[PerConditionPredict] model_folder={model_folder}\n"
        f"  checkpoint={args.chk}\n"
        f"  total_conditions={len(condition_labels)}\n"
        f"  selected_labels={selected_labels}"
    )

    base_network = predictor.network
    num_output_channels = int(predictor.label_manager.num_segmentation_heads)
    label_to_idx = {int(v): i for i, v in enumerate(condition_labels)}

    for label_value in selected_labels:
        cond_idx = int(label_to_idx[int(label_value)])
        out_dir = join(args.o, f"label_{int(label_value):02d}")
        maybe_mkdir_p(out_dir)
        print(f"\n[PerConditionPredict] Running label={label_value} (condition_index={cond_idx}) -> {out_dir}")

        predictor.network = FixedConditionBinaryAsMulticlass(
            base_network=base_network,
            condition_index=cond_idx,
            class_label=int(label_value),
            num_output_channels=num_output_channels,
        )
        predictor.predict_from_files(
            args.i,
            out_dir,
            save_probabilities=False,
            overwrite=not args.continue_prediction,
            num_processes_preprocessing=args.npp,
            num_processes_segmentation_export=args.nps,
            folder_with_segs_from_prev_stage=None,
            num_parts=args.num_parts,
            part_id=args.part_id,
        )

    predictor.network = base_network
    print("\n[PerConditionPredict] Done.")


if __name__ == "__main__":
    main()
