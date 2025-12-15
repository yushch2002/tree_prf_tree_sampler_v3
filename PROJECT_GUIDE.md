# Instructor Guide — v3

## What is "stronger" about v3?
v3 adds three enhancements commonly described for tree-structured sampling:

1) **Semantic (K-means) tree**
   Instead of an arbitrary balanced tree, we split tokens by clustering their embeddings.
   Intuition: tokens that "behave similarly" (in embedding space) tend to have similar scores,
   so subtree mass comparisons can become less noisy.

2) **Balance constraint**
   Each split is forced to keep left/right subtree sizes within `[BALANCE_MIN, BALANCE_MAX]`.
   This guarantees tree depth remains O(log V), preventing degenerate trees.

3) **L repeated traversals**
   For a query `h`, we can traverse the tree `L` times and aggregate results:
   - `mode`: pick the most frequent token among L samples (low-variance "best guess")
   - `resample`: sample from the empirical histogram (keeps stochasticity)
   This can reduce variance in the final output compared to a single traversal.

## Trade-offs
- Semantic tree is **slower to build** than a simple balanced pairing tree.
- Online complexity remains **O(D log V)** either way.
- On GPU, baseline softmax may be faster; the asymptotic benefit is best shown on CPU.

## Files
- `config.py`: defaults
- `src/kernels.py`: PRF mapping (positive features)
- `src/kmeans2.py`: fast-ish 2-means (torch) + balance enforcement + fallbacks
- `src/vocab_tree.py`: tree nodes + sampling walk
- `src/tree_builders.py`: balanced builder + semantic (k-means) builder
- `src/samplers.py`: baseline and TreePRF samplers; includes L-traversal output strategies
- `src/metrics.py`: JS divergence + top-k overlap
- `main.py`: runs benchmarks and prints instructor-facing summaries
- `tests/test_v3.py`: correctness & balance tests

## How to present results
1) `python -m pytest -q` (correctness)
2) Run speed demo with `--tree balanced` (clear speedup)
3) Run semantic demo with `--tree semantic --L 20 --L_strategy mode` and compare quality metrics
