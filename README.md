# Tree-Structured Accelerated Sampling with PRF — v3 (Semantic K-means Tree + Balance + L-Traversals)

This version adds the extensions you asked for:
- **Semantic tree** built with **bisecting 2-means** over embeddings
- **Balance constraint** per split (ratio in `[BALANCE_MIN, BALANCE_MAX]`)
- **L repeated traversals** to produce the final output token (mode or resample)

## Install
```bash
pip install -r requirements.txt
```

## Quick sanity (tests)
```bash
python -m pytest -q
```

## Run demos

### 1) Recommended instructor demo (CPU, large vocab, balanced tree for speed)
This shows the *asymptotic* speed advantage cleanly:
```bash
python main.py --device cpu --tree balanced --normalize_inputs --vocab 200000 --d_model 256 --D 256 --trials 3000 --quality_vocab 5000
```

### 2) Semantic tree + L traversals, resample
```bash
python main.py --device cpu --tree semantic --normalize_inputs \
  --vocab 50000 --d_model 256 --D 256 --trials 2000 --quality_vocab 5000 \
  --kmeans_iters 10 --L 20 --L_strategy resample

```

Notes:
- On **GPU**, baseline softmax (GEMV) can be extremely fast; tree sampling is branchy.
- Semantic K-means tree may improve quality/top-k overlap, but has higher **offline** cost.
