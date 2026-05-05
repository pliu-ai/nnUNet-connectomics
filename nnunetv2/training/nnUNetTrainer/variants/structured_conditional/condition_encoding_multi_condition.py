from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class GroupConditionEncoderMulti(nn.Module):
    """
    Encodes either:
    - group ids: [B]
    - multi-hot group mask: [B, num_groups]

    into one condition vector per sample.
    """

    def __init__(self, num_groups: int, embedding_dim: int = 64, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        self.num_groups = int(num_groups)
        self.embedding_dim = int(embedding_dim)
        hidden = int(hidden_dim) if hidden_dim is not None else int(embedding_dim)

        if self.num_groups <= 0:
            raise ValueError("num_groups must be > 0")

        self.embedding = nn.Embedding(self.num_groups, self.embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
        )
        self.output_dim = hidden

    def _normalize_condition(
        self,
        condition: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not torch.is_tensor(condition):
            condition = torch.as_tensor(condition, device=device)
        condition = condition.to(device=device)

        if condition.ndim == 2:
            if condition.shape[1] != self.num_groups:
                raise ValueError(
                    f"condition width mismatch: got {condition.shape[1]}, expected {self.num_groups}"
                )
            if condition.shape[0] == 1 and batch_size > 1:
                condition = condition.expand(batch_size, -1)
            if condition.shape[0] != batch_size:
                raise ValueError(f"condition batch mismatch: got {condition.shape[0]}, expected {batch_size}")

            cond_mask = (condition > 0).float()
            empty = cond_mask.sum(dim=1) <= 0
            if empty.any():
                cond_mask = cond_mask.clone()
                cond_mask[empty, 0] = 1.0
            cond_mask = cond_mask / cond_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            return cond_mask

        cond_ids = condition.reshape(-1).long()
        if cond_ids.numel() == 1 and batch_size > 1:
            cond_ids = cond_ids.expand(batch_size)
        if cond_ids.numel() != batch_size:
            raise ValueError(f"condition batch mismatch: got {cond_ids.numel()}, expected {batch_size}")
        return cond_ids.clamp(min=0, max=self.num_groups - 1)

    def forward(self, condition: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        cond_repr = self._normalize_condition(condition, batch_size=batch_size, device=device)
        if cond_repr.ndim == 2:
            emb = torch.matmul(cond_repr, self.embedding.weight)
        else:
            emb = self.embedding(cond_repr)
        return self.mlp(emb)


class FiLMModulation(nn.Module):
    """
    Feature-wise linear modulation (FiLM): y = gamma * x + beta.

    The affine layer is initialized as identity so conditional modulation starts
    from a stable no-op behavior.
    """

    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.affine = nn.Linear(int(cond_dim), 2 * self.channels)

        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)
        with torch.no_grad():
            self.affine.bias[: self.channels].fill_(1.0)

    def forward(self, x: torch.Tensor, cond_vector: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.affine(cond_vector)
        gamma, beta = torch.split(gamma_beta, self.channels, dim=1)
        view_shape = [x.shape[0], self.channels] + [1] * (x.ndim - 2)
        gamma = gamma.view(*view_shape)
        beta = beta.view(*view_shape)
        return gamma * x + beta
