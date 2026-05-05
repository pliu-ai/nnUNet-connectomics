from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class GroupConditionEncoder(nn.Module):
    """
    Converts an integer dynamic-group ID into a learnable condition vector.

    The output vector is used by FiLM modules in the decoder/head side.
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

    def _normalize_group_ids(
        self,
        group_ids: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not torch.is_tensor(group_ids):
            group_ids = torch.as_tensor(group_ids, device=device)
        group_ids = group_ids.to(device=device).reshape(-1).long()

        if group_ids.numel() == 1 and batch_size > 1:
            group_ids = group_ids.expand(batch_size)
        if group_ids.numel() != batch_size:
            raise ValueError(f"group_ids batch mismatch: got {group_ids.numel()}, expected {batch_size}")
        return group_ids.clamp(min=0, max=self.num_groups - 1)

    def forward(self, group_ids: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        group_ids = self._normalize_group_ids(group_ids, batch_size=batch_size, device=device)
        emb = self.embedding(group_ids)
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

        # Identity initialization: gamma = 1, beta = 0.
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
