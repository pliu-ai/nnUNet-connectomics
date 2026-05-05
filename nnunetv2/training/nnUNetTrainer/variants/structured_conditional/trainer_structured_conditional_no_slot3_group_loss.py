from __future__ import annotations

import os

import torch

from .structured_loss_no_slot3_group import StructuredConditionalLoss, StructuredLossConfig
from .trainer_structured_conditional_no_slot3 import nnUNetTrainerStructuredConditionalNoSlot3


class nnUNetTrainerStructuredConditionalNoSlot3GroupLoss(nnUNetTrainerStructuredConditionalNoSlot3):
    """
    no_slot3 structured conditional trainer with an additional group-level Dice loss.

    Group auxiliary loss for each selected dynamic group:
        gt_group = gt_cond_slot_1 OR gt_cond_slot_2
        p_group = p_cond_slot_1 + p_cond_slot_2
        L_group = Dice(p_group, gt_group)

    Total loss:
        L = CE + Dice_all + lambda_cond * Dice_slots
            + lambda_group * Dice_group + lambda_suppress * Suppression
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.loss_cfg = StructuredLossConfig(
            lambda_ce=float(os.environ.get("NNUNET_STRUCTCOND_LAMBDA_CE", "1.0")),
            lambda_dice=float(os.environ.get("NNUNET_STRUCTCOND_LAMBDA_DICE", "1.0")),
            lambda_cond=float(os.environ.get("NNUNET_STRUCTCOND_LAMBDA_COND", "0.25")),
            lambda_group=float(os.environ.get("NNUNET_STRUCTCOND_LAMBDA_GROUP", "0.4")),
            lambda_suppress=float(os.environ.get("NNUNET_STRUCTCOND_LAMBDA_SUPPRESS", "0.1")),
            enable_conditional_focus=str(os.environ.get("NNUNET_STRUCTCOND_ENABLE_COND", "1")).lower()
            in {"1", "true", "yes", "y"},
            enable_group_focus=str(os.environ.get("NNUNET_STRUCTCOND_ENABLE_GROUP", "1")).lower()
            in {"1", "true", "yes", "y"},
            enable_suppression=str(os.environ.get("NNUNET_STRUCTCOND_ENABLE_SUPPRESS", "1")).lower()
            in {"1", "true", "yes", "y"},
            batch_dice=self.configuration_manager.batch_dice,
            smooth=1e-5,
            ddp=self.is_ddp,
        )

    def _build_loss(self):
        return StructuredConditionalLoss(self.loss_cfg)
