from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal
import torch

from .kernels import PositiveRandomFeatures
from .tree_builders import build_tree_balanced, build_tree_semantic
from .vocab_tree import VocabTree

LStrategy = Literal["mode", "resample", "first"]

@dataclass
class BaselineSoftmaxSampler:
    E: torch.Tensor
    b: Optional[torch.Tensor]
    temperature: float = 1.0

    @torch.no_grad()
    def sample(self, h: torch.Tensor, rng: torch.Generator) -> int:
        T = float(self.temperature)
        h = h.to(dtype=torch.float64, device=self.E.device)
        logits = self.E @ (h / T)
        if self.b is not None:
            logits = logits + (self.b / T)
        probs = torch.softmax(logits, dim=0)
        return int(torch.multinomial(probs, 1, True, generator=rng).item())

@dataclass
class TreePRFSampler:
    prf: PositiveRandomFeatures
    tree: VocabTree
    temperature: float
    device: str
    eps: float
    L: int
    L_strategy: str

    @staticmethod
    @torch.no_grad()
    def preprocess(E: torch.Tensor,
                   b: Optional[torch.Tensor],
                   D: int,
                   temperature: float,
                   device: str,
                   seed: int,
                   eps: float,
                   tree_type: str,
                   embeddings_for_tree: Optional[torch.Tensor] = None,
                   balance_min: float = 0.4,
                   balance_max: float = 0.6,
                   kmeans_iters: int = 10,
                   L: int = 1,
                   L_strategy: str = "mode") -> "TreePRFSampler":
        E = E.to(device=device, dtype=torch.float64)
        V, d = E.shape
        T = float(temperature)

        # W scale helps conditioning when E/h are normalized
        w_scale = 1.0 / (d ** 0.5)
        prf = PositiveRandomFeatures(input_dim=d, num_features=D, device=device, seed=seed, w_scale=w_scale)

        phiE = prf.phi(E / T)  # (V,D) positive
        if b is None:
            v = phiE
        else:
            b = b.to(device=device, dtype=torch.float64)
            v = phiE * torch.exp((b / T).unsqueeze(1))

        g = torch.Generator(device=device)
        g.manual_seed(seed)

        if tree_type == "balanced":
            tree = build_tree_balanced(v)
        elif tree_type == "semantic":
            assert embeddings_for_tree is not None, "semantic tree requires embeddings_for_tree"
            tree = build_tree_semantic(vocab_feat=v,
                                       embeddings=embeddings_for_tree.to(device=device, dtype=torch.float64),
                                       balance_min=balance_min,
                                       balance_max=balance_max,
                                       kmeans_iters=kmeans_iters,
                                       rng=g)
        else:
            raise ValueError(f"Unknown tree_type: {tree_type}")

        return TreePRFSampler(prf=prf, tree=tree, temperature=T, device=device, eps=eps, L=int(L), L_strategy=L_strategy)

    @torch.no_grad()
    def sample(self, h: torch.Tensor, rng: torch.Generator) -> int:
        h = h.to(device=self.device, dtype=torch.float64)
        u = self.prf.phi(h / float(self.temperature))

        L = max(1, int(self.L))
        if L == 1:
            return self.tree.sample_once(u, rng=rng, eps=self.eps)

        # repeated traversals
        ids = [self.tree.sample_once(u, rng=rng, eps=self.eps) for _ in range(L)]

        if self.L_strategy == "first":
            return int(ids[0])

        # histogram
        # use torch bincount for speed
        V = max(ids) + 1
        counts = torch.bincount(torch.tensor(ids, device=self.device, dtype=torch.long), minlength=V).to(dtype=torch.float64)

        if self.L_strategy == "mode":
            return int(torch.argmax(counts).item())

        # resample from empirical distribution
        probs = counts / (counts.sum() + self.eps)
        return int(torch.multinomial(probs, 1, True, generator=rng).item())
