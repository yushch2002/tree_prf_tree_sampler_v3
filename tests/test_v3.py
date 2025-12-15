import torch
from src.kmeans2 import kmeans2_split
from src.tree_builders import build_tree_semantic, build_tree_balanced
from src.kernels import PositiveRandomFeatures

def test_kmeans_balance_constraint():
    # Two obvious clusters
    torch.manual_seed(0)
    V = 200
    d = 8
    x1 = torch.randn(V//2, d) + 5.0
    x2 = torch.randn(V//2, d) - 5.0
    emb = torch.cat([x1, x2], dim=0).to(dtype=torch.float64)

    idxs = list(range(V))
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    split = kmeans2_split(idxs, emb, iters=5, balance_min=0.4, balance_max=0.6, rng=g)
    r = len(split.left) / V
    assert 0.4 <= r <= 0.6

def test_tree_mass_conservation_semantic():
    torch.manual_seed(0)
    V, D, d = 257, 16, 8
    v = torch.rand(V, D, dtype=torch.float64)
    emb = torch.randn(V, d, dtype=torch.float64)
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    tree = build_tree_semantic(v, emb, balance_min=0.4, balance_max=0.6, kmeans_iters=3, rng=g)
    root_sum = tree.nodes[tree.root].sum_feat
    assert torch.allclose(root_sum, v.sum(dim=0))

def test_prf_positive():
    prf = PositiveRandomFeatures(8, 32, device="cpu", seed=0, w_scale=1.0)
    x = torch.ones(8, dtype=torch.float64)
    phi = prf.phi(x)
    assert torch.all(phi > 0).item()
