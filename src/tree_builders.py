from __future__ import annotations
from typing import List, Optional
import torch

from .vocab_tree import Node, VocabTree
from .kmeans2 import kmeans2_split

@torch.no_grad()
def build_tree_balanced(vocab_feat: torch.Tensor) -> VocabTree:
    '''
    Bottom-up pairing build. O(V*D).
    '''
    V, D = vocab_feat.shape
    nodes: List[Node] = []
    level: List[int] = []
    for i in range(V):
        idx = len(nodes)
        nodes.append(Node(None, None, i, vocab_feat[i].clone()))
        level.append(idx)
    while len(level) > 1:
        nxt: List[int] = []
        j = 0
        while j < len(level):
            if j + 1 == len(level):
                nxt.append(level[j]); j += 1; continue
            li, ri = level[j], level[j+1]
            idx = len(nodes)
            nodes.append(Node(li, ri, None, nodes[li].sum_feat + nodes[ri].sum_feat))
            nxt.append(idx)
            j += 2
        level = nxt
    return VocabTree(nodes, level[0])

@torch.no_grad()
def build_tree_semantic(vocab_feat: torch.Tensor,
                        embeddings: torch.Tensor,
                        balance_min: float,
                        balance_max: float,
                        kmeans_iters: int,
                        rng: torch.Generator) -> VocabTree:
    '''
    Recursive bisecting 2-means tree over embeddings, storing sum_feat over vocab_feat.

    Offline cost is higher than balanced pairing; use moderate V for demos.
    '''
    V, D = vocab_feat.shape
    nodes: List[Node] = []

    def rec(indices: List[int]) -> int:
        # sum feature
        sum_feat = vocab_feat[indices].sum(dim=0)
        if len(indices) == 1:
            idx = len(nodes)
            nodes.append(Node(None, None, indices[0], sum_feat))
            return idx

        split = kmeans2_split(indices=indices,
                              embeddings=embeddings,
                              iters=kmeans_iters,
                              balance_min=balance_min,
                              balance_max=balance_max,
                              rng=rng)

        # Safety fallback if split failed
        if len(split.left) == 0 or len(split.right) == 0:
            mid = len(indices) // 2
            split_left = indices[:mid]
            split_right = indices[mid:]
        else:
            split_left, split_right = split.left, split.right

        left_idx = rec(split_left)
        right_idx = rec(split_right)

        idx = len(nodes)
        nodes.append(Node(left_idx, right_idx, None, nodes[left_idx].sum_feat + nodes[right_idx].sum_feat))
        return idx

    root = rec(list(range(V)))
    return VocabTree(nodes, root)
