import argparse
import time
import torch

from config import Config
from src.samplers import BaselineSoftmaxSampler, TreePRFSampler
from src.metrics import js_divergence, topk_overlap

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def l2_normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    n = torch.norm(x, dim=-1, keepdim=True).clamp_min(eps)
    return x / n

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=Config.DEVICE, choices=["cpu","cuda"])
    ap.add_argument("--tree", type=str, default="semantic", choices=["semantic","balanced"])
    ap.add_argument("--vocab", type=int, default=Config.VOCAB_SIZE)
    ap.add_argument("--d_model", type=int, default=Config.D_MODEL)
    ap.add_argument("--D", type=int, default=Config.PRF_DIM)
    ap.add_argument("--temperature", type=float, default=Config.TEMPERATURE)
    ap.add_argument("--trials", type=int, default=Config.TRIALS)
    ap.add_argument("--quality_vocab", type=int, default=Config.QUALITY_VOCAB)
    ap.add_argument("--seed", type=int, default=Config.SEED)
    ap.add_argument("--normalize_inputs", action="store_true")
    ap.add_argument("--balance_min", type=float, default=Config.BALANCE_MIN)
    ap.add_argument("--balance_max", type=float, default=Config.BALANCE_MAX)
    ap.add_argument("--kmeans_iters", type=int, default=Config.KMEANS_ITERS)
    ap.add_argument("--L", type=int, default=Config.L_TRAVERSALS)
    ap.add_argument("--L_strategy", type=str, default=Config.L_STRATEGY, choices=["mode","resample","first"])
    args = ap.parse_args()

    set_seed(args.seed)
    device = args.device

    V, d = int(args.vocab), int(args.d_model)
    D = int(args.D)
    T = float(args.temperature)

    print(f"DEVICE={device}")
    print(f"Tree={args.tree} | V={V}, d_model={d}, PRF D={D}, T={T}")
    print(f"Trials={args.trials}, Quality vocab={args.quality_vocab}")
    print(f"balance=[{args.balance_min},{args.balance_max}], kmeans_iters={args.kmeans_iters}")
    print(f"L={args.L}, L_strategy={args.L_strategy}, normalize_inputs={args.normalize_inputs}")
    print("")

    # Simulated decoder-head weights and hidden state
    E = torch.randn(V, d, dtype=torch.float64, device=device) / (d ** 0.5)
    b = 0.1 * torch.randn(V, dtype=torch.float64, device=device)
    h = torch.randn(d, dtype=torch.float64, device=device)

    if args.normalize_inputs:
        E = l2_normalize_rows(E, eps=Config.EPS)
        h = h / torch.norm(h).clamp_min(Config.EPS)

    # Use E as the embedding space for semantic clustering
    embeddings_for_tree = E

    g = torch.Generator(device=device)
    g.manual_seed(args.seed)

    baseline = BaselineSoftmaxSampler(E=E, b=b, temperature=T)

    # Preprocess tree sampler
    t0 = time.perf_counter()
    tree_sampler = TreePRFSampler.preprocess(E=E, b=b, D=D, temperature=T, device=device,
                                             seed=args.seed, eps=Config.EPS,
                                             tree_type=args.tree, embeddings_for_tree=embeddings_for_tree,
                                             balance_min=args.balance_min, balance_max=args.balance_max,
                                             kmeans_iters=args.kmeans_iters,
                                             L=args.L, L_strategy=args.L_strategy)
    t1 = time.perf_counter()

    print("[Preprocess]")
    print(f"PRF(vocab)+tree build time: {t1 - t0:.4f} s (one-time)")
    print("")

    # Warm-up
    for _ in range(50):
        baseline.sample(h, g)
        tree_sampler.sample(h, g)

    # Speed benchmark
    t2 = time.perf_counter()
    for _ in range(args.trials):
        baseline.sample(h, g)
    t3 = time.perf_counter()

    t4 = time.perf_counter()
    for _ in range(args.trials):
        tree_sampler.sample(h, g)
    t5 = time.perf_counter()

    baseline_ms = 1000.0 * (t3 - t2) / args.trials
    tree_ms = 1000.0 * (t5 - t4) / args.trials

    print("[Speed]")
    print(f"Baseline softmax: {baseline_ms:.6f} ms/sample")
    print(f"Tree PRF sampler: {tree_ms:.6f} ms/sample")
    print(f"Speedup:          {baseline_ms / tree_ms:.2f}x")
    if device == "cuda" and baseline_ms / tree_ms < 1.0:
        print("Note: On GPU, baseline GEMV is highly optimized; tree is branchy. For speedup, use CPU or very large vocab.")
    print("")

    # Quality on small subset (exact compare)
    Vq = min(int(args.quality_vocab), V)
    Eq = E[:Vq].contiguous()
    bq = b[:Vq].contiguous()

    logits = (Eq @ (h / T)) + (bq / T)
    p_exact = torch.softmax(logits, dim=0)

    # Approx distribution for same subset induced by PRF (direct compute, not via tree)
    prf = tree_sampler.prf
    u = prf.phi(h / T)
    phiEq = prf.phi(Eq / T)
    w = (phiEq @ u) * torch.exp(bq / T)
    p_approx = w / (w.sum() + Config.EPS)

    print("[Quality] (exact vs PRF-approx on small vocab)")
    print(f"JS divergence:  {js_divergence(p_exact, p_approx, eps=Config.EPS):.6f}")
    print(f"Top-50 overlap: {topk_overlap(p_exact, p_approx, k=min(50, Vq)):.3f}")
    print("")

    # Empirical check on SAME small vocab, using the SAME tree type and L
    baseline_q = BaselineSoftmaxSampler(E=Eq, b=bq, temperature=T)
    tree_q = TreePRFSampler.preprocess(E=Eq, b=bq, D=D, temperature=T, device=device,
                                       seed=args.seed, eps=Config.EPS,
                                       tree_type=args.tree, embeddings_for_tree=Eq,
                                       balance_min=args.balance_min, balance_max=args.balance_max,
                                       kmeans_iters=args.kmeans_iters,
                                       L=args.L, L_strategy=args.L_strategy)

    N_emp = 20000
    counts_base = torch.zeros(Vq, dtype=torch.float64, device=device)
    counts_tree = torch.zeros(Vq, dtype=torch.float64, device=device)

    for _ in range(N_emp):
        i = baseline_q.sample(h, g)
        counts_base[i] += 1
        j = tree_q.sample(h, g)
        if j < Vq:
            counts_tree[j] += 1

    freq_base = counts_base / (counts_base.sum() + Config.EPS)
    freq_tree = counts_tree / (counts_tree.sum() + Config.EPS)

    print("[Empirical check] (sampling frequencies on the SAME small vocab)")
    print(f"JS(p_exact, freq_baseline): {js_divergence(p_exact, freq_base, eps=Config.EPS):.6f}")
    print(f"JS(p_exact, freq_tree):     {js_divergence(p_exact, freq_tree, eps=Config.EPS):.6f}")
    print("")
    print("How to explain to the instructor:")
    print("- Baseline is O(V): computes all logits and softmax each sample.")
    print("- Tree sampler is O(D log V): compute u=phi(h) then walk ~log2(V) nodes, each with one dot(u, sum_feat).")
    print("- Semantic tree uses 2-means splits + balance constraints to keep depth ~log V.")
    print("- L-traversals reduces variance: mode over L samples is a low-variance 'best guess' output.")

if __name__ == "__main__":
    main()
