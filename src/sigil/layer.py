# src/sigil/layer.py
import math
import torch
import torch.nn as nn

class SigilLayer(nn.Module):
    """
    SigilLayer: wraps a base torch layer and maintains a per-output 'phi' vector
    (Value-Weighted Pathway Reinforcement). This file provides a minimal,
    stable update_phi() to link phi -> real gradient activity (VWPR Stage 1).
    """

    def __init__(self, base_layer, phi_init: float = 1.0, phi_min: float = 1e-6, phi_max: float = 1e3):
        super().__init__()
        self.base_layer = base_layer

        # infer number of outputs for phi
        out_features = getattr(base_layer, "out_features", None)
        if out_features is None:
            out_features = getattr(base_layer, "out_channels", None)
        if out_features is None:
            out_features = 64  # safe fallback

        # phi param (we update manually with update_phi())
        self.phi = nn.Parameter(torch.full((out_features,), float(phi_init)), requires_grad=False)

        # momentum buffer for phi updates (registered buffer, not a parameter)
        self.register_buffer("_phi_momentum", torch.zeros_like(self.phi))

        # history snapshots (kept on CPU to avoid bloating GPU memory)
        self.phi_history = []

        # metadata and clamps
        self.name = getattr(base_layer, "name", base_layer.__class__.__name__)
        self.phi_min = float(phi_min)
        self.phi_max = float(phi_max)

    def forward(self, x):
        # NB: Stage 1 does not apply phi to the forward pass.
        return self.base_layer(x)

    def update_phi(self,
                   lr_phi: float = 1e-4,
                   momentum: float = 0.9,
                   clip_std: float = 3.0,
                   verbose: bool = False):
        """
        Minimal VWPR update (Stage 1).
        - lr_phi: small learning rate for phi updates (default 1e-4)
        - momentum: momentum for internal phi buffer (0..1)
        - clip_std: after update, clamp phi to [mean-clip_std*std, mean+clip_std*std] for stability
        - verbose: print a small line for debugging

        Returns:
            dict with simple diagnostics {"phi_mean", "phi_std"}
        """
        with torch.no_grad():
            device = self.phi.device

            # 1) Derive per-output gradient signal
            weight = getattr(self.base_layer, "weight", None)
            if (weight is not None) and (weight.grad is not None):
                g = weight.grad.detach()
                if g.ndim >= 2:
                    # For linear: (out, in) -> norm over input dims => per-out
                    reduce_dims = tuple(range(1, g.ndim))
                    per_out = torch.norm(g, dim=reduce_dims).view(-1).to(device)
                else:
                    per_out = g.detach().view(-1).to(device)
            else:
                bias = getattr(self.base_layer, "bias", None)
                if (bias is not None) and (bias.grad is not None):
                    per_out = bias.grad.detach().view(-1).to(device)
                else:
                    # fallback tiny noise (keeps phi moving slightly if no grads present)
                    per_out = torch.randn_like(self.phi) * 1e-8

            # 2) Normalize per_out to [0,1] robustly
            per_out = per_out.float()
            minv = float(per_out.min().item())
            maxv = float(per_out.max().item())
            denom = (maxv - minv)
            if denom < 1e-12:
                # all-equal or near-zero: produce a tiny, stable signal
                per_out_norm = torch.zeros_like(per_out)
            else:
                per_out_norm = (per_out - minv) / denom

            # small jitter if constant
            if torch.allclose(per_out_norm, per_out_norm[0]):
                per_out_norm = per_out_norm + (torch.randn_like(per_out_norm) * 1e-9)

            # 3) Compute delta and apply momentum
            delta = lr_phi * per_out_norm  # small positive increments
            # update momentum buffer: _phi_momentum <- momentum*_phi_momentum + (1-momentum)*delta
            self._phi_momentum.mul_(momentum).add_(delta * (1.0 - momentum))

            # apply momentum-driven delta to phi
            self.phi.data.add_(self._phi_momentum)

            # 4) Clamp phi for numeric stability
            mean = float(self.phi.data.mean().item())
            std = float(self.phi.data.std().item()) + 1e-12
            lower = max(self.phi_min, mean - clip_std * std)
            upper = min(self.phi_max, mean + clip_std * std)
            self.phi.data.clamp_(min=lower, max=upper)
            # also ensure global min/max
            self.phi.data.clamp_(min=self.phi_min, max=self.phi_max)

            # Do NOT record epoch here — let manager.record_epoch() handle snapshots.
            if verbose:
                print(f"[SigilLayer:{self.name}] phi_mean={self.phi.data.mean():.6f} phi_std={self.phi.data.std():.6f}")

            return {"phi_mean": float(self.phi.data.mean().item()), "phi_std": float(self.phi.data.std().item())}

    def record_epoch(self):
        """Store a CPU snapshot of the current phi for historical metrics."""
        self.phi_history.append(self.phi.detach().cpu().clone())

    def get_phi(self):
        return self.phi

