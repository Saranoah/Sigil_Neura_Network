import torch
import torch.nn as nn
from sigil.layer import SigilLayer
from sigil.manager import SigilNetworkManager

# -------------------------
# Build a sample network
# -------------------------
model = nn.Sequential(
    SigilLayer(nn.Linear(128, 64)),
    SigilLayer(nn.ReLU()),
    SigilLayer(nn.Linear(64, 32)),
    SigilLayer(nn.ReLU()),
    SigilLayer(nn.Linear(32, 10))
)

manager = SigilNetworkManager(model)

# -------------------------
# Simulated training loop
# -------------------------
for epoch in range(50):
    # Placeholder for actual training logic
    for layer in manager.sigil_layers:
        layer.update_phi()  # update phi after each epoch
    manager.record_epoch()

    if epoch % 10 == 0:
        print(f"Generating gallery for epoch {epoch}...")
        manager.generate_gallery(epoch)

# -------------------------
# Print final metrics
# -------------------------
print("Final Network Metrics:")
print(manager.summary_metrics())

# Copyright (c) 2024 Israa Ali
# Licensed under the MIT License
