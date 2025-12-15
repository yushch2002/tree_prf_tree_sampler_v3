from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import torch

@dataclass
class Node:
    left: Optional[int]
    right: Optional[int]
    token_id: Optional[int]
    sum_feat: torch.Tensor

class VocabTree:
    def __init__(self, nodes: List[Node], root: int):
        self.nodes = nodes
        self.root = root

    @torch.no_grad()
    def sample_once(self, u: torch.Tensor, rng: torch.Generator, eps: float = 1e-12) -> int:
        '''
        Sample 1 token by mass-splitting walk:
          r ~ Uniform(0, mass(root))
          go left if r < mass(left); else r -= mass(left) and go right.
        '''
        u = u.to(dtype=torch.float64, device=self.nodes[self.root].sum_feat.device)
        idx = self.root
        total = torch.dot(u, self.nodes[idx].sum_feat).clamp_min(eps)
        r = torch.rand((), generator=rng, device=total.device, dtype=torch.float64) * total

        while True:
            node = self.nodes[idx]
            if node.token_id is not None:
                return int(node.token_id)
            left = self.nodes[node.left]
            left_mass = torch.dot(u, left.sum_feat).clamp_min(eps)
            if r < left_mass:
                idx = node.left
            else:
                r = r - left_mass
                idx = node.right
