import torch
from torch import nn
import torch.nn.functional as F

from .math_utils import pairwise_squared_distance


class SinkhornTransport(nn.Module):
    def __init__(self, alpha, init_source=None, init_target=None, max_iter=5000, stop_threshold=0.5e-2):
        super().__init__()
        self.alpha = alpha
        self.max_iter = max_iter
        self.stop_threshold = stop_threshold
        self.init_source = init_source
        self.init_target = init_target
        if init_source is not None:
            self.source_dist = init_source
        if init_target is not None:
            self.target_dist = init_target

    def forward(self, x, y):
        cost = pairwise_squared_distance(x, y)
        device = cost.device
        source = self._distribution(self.init_source, getattr(self, "source_dist", None), cost.shape[0], device)
        target = self._distribution(self.init_target, getattr(self, "target_dist", None), cost.shape[1], device)
        log_source = torch.log(source + 1e-30)
        log_target = torch.log(target + 1e-30)
        log_u = torch.zeros_like(log_source)
        log_v = torch.zeros_like(log_target)
        log_kernel = -cost * self.alpha
        error = 1.0
        step = 0
        while error > self.stop_threshold and step < self.max_iter:
            log_v = log_target - torch.logsumexp(log_kernel.T + log_u.T, dim=1).unsqueeze(1)
            log_u = log_source - torch.logsumexp(log_kernel + log_v.T, dim=1).unsqueeze(1)
            step += 1
            if step % 50 == 1:
                log_kernel = log_kernel + log_u + log_v.T
                log_u = torch.zeros_like(log_source)
                log_v = torch.zeros_like(log_target)
                error = self._marginal_error(log_kernel, log_u, log_v, source, target)
        transport = torch.exp(log_u) * (torch.exp(log_kernel) * torch.exp(log_v).T)
        loss = torch.sum(transport * cost)
        return loss, transport

    def _distribution(self, init_flag, logits, size, device):
        if init_flag is None:
            return (torch.ones(size, device=device) / size).unsqueeze(1)
        return F.softmax(logits, dim=0).to(device)

    def _marginal_error(self, log_kernel, log_u, log_v, source, target):
        with torch.no_grad():
            row_sums = torch.exp(log_u + torch.logsumexp(log_kernel + log_v.T, dim=1, keepdim=True))
            col_sums = torch.exp(log_v + torch.logsumexp(log_kernel.T + log_u.T, dim=1, keepdim=True))
            row_error = torch.abs(row_sums - source).sum()
            col_error = torch.abs(col_sums - target).sum()
            return max(row_error.item(), col_error.item())
