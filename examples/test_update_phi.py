# examples/test_update_phi.py
import torch
import torch.nn as nn
from sigil.layer import SigilLayer
from sigil.manager import SigilNetworkManager

# build model using SigilLayer wrappers
model = nn.Sequential(
    SigilLayer(nn.Linear(128, 64)),
    SigilLayer(nn.ReLU()),
    SigilLayer(nn.Linear(64, 32)),
    SigilLayer(nn.ReLU()),
    SigilLayer(nn.Linear(32, 10))
)

manager = SigilNetworkManager(model)

# Simulate a few epochs; set fake gradients so update_phi uses them.
for epoch in range(5):
    # simulate weight gradients for layers that have weights
    for layer in manager.sigil_layers:
        w = getattr(layer.base_layer, "weight", None)
        if w is not None:
            # simulate a plausible small gradient
            w.grad = torch.randn_like(w) * (1e-2 + 1e-3 * epoch)
        b = getattr(layer.base_layer, "bias", None)
        if b is not None:
            b.grad = torch.randn_like(b) * (1e-3 + 1e-4 * epoch)

    # call update_phi for each layer (Stage 1)
    for layer in manager.sigil_layers:
        metrics = layer.update_phi(lr_phi=1e-4, momentum=0.9)
        print(f"Epoch {epoch:02d} | Layer {layer.name:20} | mean={metrics['phi_mean']:.6f} std={metrics['phi_std']:.6f}")

    # manager records snapshot if you want
    manager.record_epoch()
