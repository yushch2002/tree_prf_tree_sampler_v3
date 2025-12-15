import math
import torch

class PositiveRandomFeatures:
    '''
    Positive Random Features (PRF) mapping.

    We use:
      a = W x - ||x||^2/2
      phi(x) = exp(a) / sqrt(D)

    Notes:
    - phi(x) is strictly positive => subtree masses are nonnegative.
    - This mapping is a practical FAVOR+-style feature map used to approximate exp(x^T y).
    '''
    def __init__(self, input_dim: int, num_features: int, device: str, seed: int = 0, w_scale: float = 1.0):
        self.input_dim = int(input_dim)
        self.D = int(num_features)
        self.device = device

        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        W = torch.randn(self.D, self.input_dim, generator=g, dtype=torch.float64) * float(w_scale)
        self.W = W.to(device)

    @torch.no_grad()
    def phi(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(dtype=torch.float64, device=self.device)
        proj = x @ self.W.t()                       # (B,D)
        x_norm_sq_half = 0.5 * (x * x).sum(dim=-1, keepdim=True)  # (B,1)
        a = proj - x_norm_sq_half
        out = torch.exp(a) / math.sqrt(self.D)
        return out.squeeze(0) if out.shape[0] == 1 else out
