import torch
import torch.nn as nn

class SigilLayer(nn.Module):
    """
    SigilLayer
    ==========
    A wrapper around standard PyTorch layers that introduces
    a secondary parameter vector `phi`, representing
    Value-Weighted Pathway Reinforcement (VWPR).
    """

    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer
        out_features = getattr(base_layer, 'out_features', 64)  # fallback
        self.phi = nn.Parameter(torch.ones(out_features))
        self.phi_history = []
        self.name = getattr(base_layer, 'name', base_layer.__class__.__name__)

    def forward(self, x):
        return self.base_layer(x)

    def update_phi(self, loss_map=None):
        """
        Placeholder update rule for phi.
        Replace this with true VWPR logic later.
        """
        self.phi.data += torch.randn_like(self.phi) * 1e-3

    def record_epoch(self):
        """Record phi state at the end of each epoch."""
        self.phi_history.append(self.phi.detach().cpu().clone())

    def get_phi(self):
        """Return current phi values."""
        return self.phi

