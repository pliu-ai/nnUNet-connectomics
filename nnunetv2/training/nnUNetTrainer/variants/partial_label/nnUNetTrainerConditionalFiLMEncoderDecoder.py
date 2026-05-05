from __future__ import annotations

from nnunetv2.training.nnUNetTrainer.variants.partial_label.nnUNetTrainerConditionalFiLM import (
    nnUNetTrainerConditionalFiLM,
)


class nnUNetTrainerConditionalFiLMEncoderDecoder(nnUNetTrainerConditionalFiLM):
    """
    Explicit trainer alias for FiLM conditioning applied in both encoder and decoder.

    Notes:
    - This trainer keeps all training/validation behavior of nnUNetTrainerConditionalFiLM.
    - The underlying network (ConditionalFiLMUNet) already injects condition in:
      1) encoder stages (after each encoder stage output)
      2) decoder stages (after each decoder stage output)
    - Created as a separate trainer name so experiments can reference this setting explicitly
      without changing existing trainer names.
    """

    pass

