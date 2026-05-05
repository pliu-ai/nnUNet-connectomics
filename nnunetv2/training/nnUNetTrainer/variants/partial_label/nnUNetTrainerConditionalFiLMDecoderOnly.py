from __future__ import annotations

from typing import List, Tuple, Union

import torch

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.conditional_film_unet_decoder_only import (
    ConditionalFiLMUNetDecoderOnly,
)
from nnunetv2.training.nnUNetTrainer.variants.partial_label.nnUNetTrainerConditionalFiLM import (
    nnUNetTrainerConditionalFiLM,
)
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


class nnUNetTrainerConditionalFiLMDecoderOnly(nnUNetTrainerConditionalFiLM):
    """
    Conditional FiLM trainer with decoder-only condition injection.

    Keeps all training/validation logic from nnUNetTrainerConditionalFiLM and
    only swaps network architecture construction.
    """

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> torch.nn.Module:
        binary_output_channels = 2
        num_conditions = max(1, int(num_output_channels) - 1)
        backbone = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            binary_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        return ConditionalFiLMUNetDecoderOnly(
            backbone=backbone,
            num_conditions=num_conditions,
            num_output_channels=int(num_output_channels),
            cond_dim=64,
        )

