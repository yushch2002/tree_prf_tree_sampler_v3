import torch

@torch.no_grad()
def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    p = p.to(dtype=torch.float64)
    q = q.to(dtype=torch.float64)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)

    def kl(a, b):
        a = torch.clamp(a, eps, 1.0)
        b = torch.clamp(b, eps, 1.0)
        return (a * (torch.log(a) - torch.log(b))).sum().item()

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

@torch.no_grad()
def topk_overlap(p: torch.Tensor, q: torch.Tensor, k: int = 50) -> float:
    kp = torch.topk(p, k=min(k, p.numel())).indices.tolist()
    kq = torch.topk(q, k=min(k, q.numel())).indices.tolist()
    return len(set(kp) & set(kq)) / float(min(k, p.numel()))
