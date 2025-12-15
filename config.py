import torch

class Config:
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    VOCAB_SIZE = 50000
    D_MODEL = 256
    PRF_DIM = 256
    TEMPERATURE = 1.0

    TRIALS = 2000
    QUALITY_VOCAB = 5000

    # v3 additions
    BALANCE_MIN = 0.4
    BALANCE_MAX = 0.6
    KMEANS_ITERS = 10
    L_TRAVERSALS = 1
    L_STRATEGY = "mode"  # mode | resample | first

    EPS = 1e-12
