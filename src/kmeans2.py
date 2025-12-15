from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
import torch

@dataclass
class SplitResult:
    left: List[int]
    right: List[int]

def _fallback_half_split(indices: List[int], x: torch.Tensor) -> SplitResult:
    # Split by random projection median (deterministic-ish with torch RNG already seeded)
    n = len(indices)
    if n <= 1:
        return SplitResult(indices, [])
    d = x.shape[1]
    r = torch.randn(d, dtype=torch.float64, device=x.device)
    proj = (x @ r).detach()
    order = torch.argsort(proj)
    mid = n // 2
    left_idx = order[:mid].tolist()
    right_idx = order[mid:].tolist()
    left = [indices[i] for i in left_idx]
    right = [indices[i] for i in right_idx]
    return SplitResult(left, right)

@torch.no_grad()
def kmeans2_split(indices: List[int],
                  embeddings: torch.Tensor,
                  iters: int,
                  balance_min: float,
                  balance_max: float,
                  rng: torch.Generator) -> SplitResult:
    '''
    Run 2-means on embeddings[indices] and return a balanced split.
    - embeddings: (V, d)
    - indices: subset indices
    - Returns: left/right token ids (original indices, not local)

    Balance enforcement:
      ratio = |left|/n must be in [balance_min, balance_max].
      If not, we move points from the larger cluster to the smaller based on distance margin.
    '''
    n = len(indices)
    if n <= 2:
        if n == 2:
            return SplitResult([indices[0]], [indices[1]])
        return SplitResult(indices, [])

    x = embeddings[indices].to(dtype=torch.float64)  # (n,d)

    # init centers: pick two random points
    perm = torch.randperm(n, generator=rng, device=x.device)
    c1 = x[perm[0]].clone()
    c2 = x[perm[1]].clone()

    for _ in range(max(1, int(iters))):
        # squared distances
        d1 = ((x - c1) ** 2).sum(dim=1)
        d2 = ((x - c2) ** 2).sum(dim=1)
        labels = (d2 < d1)  # True -> cluster 2, False -> cluster 1

        # handle empty cluster
        if labels.all() or (~labels).all():
            return _fallback_half_split(indices, x)

        c1 = x[~labels].mean(dim=0)
        c2 = x[labels].mean(dim=0)

    # Final assignment
    d1 = ((x - c1) ** 2).sum(dim=1)
    d2 = ((x - c2) ** 2).sum(dim=1)
    labels = (d2 < d1)

    left_local = torch.where(~labels)[0]
    right_local = torch.where(labels)[0]

    # Ensure non-empty
    if left_local.numel() == 0 or right_local.numel() == 0:
        return _fallback_half_split(indices, x)

    # Balance enforcement
    ratio = left_local.numel() / n
    if ratio < balance_min or ratio > balance_max:
        # Determine which side is too big
        if ratio < balance_min:
            # left too small -> move some from right to left
            need = int(torch.ceil(torch.tensor(balance_min * n - left_local.numel(), dtype=torch.float64)).item())
            big = right_local
            # candidates: those closer to c1 than c2 (small margin)
            margin = (d2 - d1)[big]  # positive => closer to c1
            # sort descending: most suitable to move (largest positive)
            order = torch.argsort(margin, descending=True)
            move = big[order[:need]]
            left_local = torch.cat([left_local, move])
            mask = torch.ones_like(right_local, dtype=torch.bool)
            # remove moved
            moved_set = set(move.tolist())
            kept = [i for i in right_local.tolist() if i not in moved_set]
            right_local = torch.tensor(kept, device=x.device, dtype=torch.long)
        else:
            # left too big -> move some from left to right
            need = int(torch.ceil(torch.tensor(left_local.numel() - balance_max * n, dtype=torch.float64)).item())
            big = left_local
            margin = (d1 - d2)[big]  # positive => closer to c2
            order = torch.argsort(margin, descending=True)
            move = big[order[:need]]
            right_local = torch.cat([right_local, move])
            moved_set = set(move.tolist())
            kept = [i for i in left_local.tolist() if i not in moved_set]
            left_local = torch.tensor(kept, device=x.device, dtype=torch.long)

        # If still degenerate, fallback
        if left_local.numel() == 0 or right_local.numel() == 0:
            return _fallback_half_split(indices, x)

    left = [indices[i] for i in left_local.tolist()]
    right = [indices[i] for i in right_local.tolist()]
    return SplitResult(left, right)
