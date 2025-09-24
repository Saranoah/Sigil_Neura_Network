# 🌌 # EnhancedVWPRWrapper

A PyTorch wrapper implementing Value-Weighted Parameter Refinement (VWPR) optimization, inspired by Kintsugi principles of learning from errors.

## Overview

The `EnhancedVWPRWrapper` transforms traditional loss minimization into a dual-stream learning process that preserves and amplifies informative error signals. Instead of simply minimizing loss, this approach identifies valuable error patterns and uses them to guide optimization.

## Key Concepts

- **Local Error Signals (l_i)**: Capture unusual or rare activations that may contain valuable information
- **Value Weights (φ_i)**: Dynamically adjust based on error magnitude to preserve informative pathways  
- **Dual Optimization**: Simultaneously updates model parameters (θ) and value weights (φ)

## Features

- **Dual-Stream Updates**: Simultaneous optimization of model weights and value weights
- **Softplus Activation**: Ensures positive value weights with smooth gradients
- **Phi Normalization**: Optional constraint to balance value weights across the network
- **L1 Regularization**: Prevents runaway value weight growth
- **Gradient Amplification**: Increases learning on high-value error pathways
- **History Tracking**: Stores value weight evolution for analysis

## Installation

```bash
pip install torch torchvision
# Add any additional dependencies
```

## Usage

### Basic Implementation

```python
import torch
import torch.nn as nn
from enhanced_vwpr_wrapper import EnhancedVWPRWrapper

# Wrap your existing model
model = YourModel()
vwpr_model = EnhancedVWPRWrapper(model)

# Setup optimizers for dual-stream learning
optimizer_theta = torch.optim.Adam(vwpr_model.base_model.parameters(), lr=0.001)
optimizer_phi = torch.optim.Adam(vwpr_model.phi_params(), lr=0.01)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        output = vwpr_model(batch)
        base_loss = loss_function(output, targets)
        
        # Apply VWPR step
        l_map, phi_vals = vwpr_model.apply_vwpr_step(
            base_loss, 
            optimizer_theta, 
            optimizer_phi
        )
        
        # Optional: visualize phi distribution
        if epoch % 10 == 0:
            vwpr_model.visualize_phi_distribution(epoch)
```

### Configuration Options

```python
vwpr_model = EnhancedVWPRWrapper(
    base_model=model,
    normalize_phi=True,        # Enable phi normalization
    l1_lambda=0.01,           # L1 regularization strength
    phi_clip_value=10.0,      # Maximum phi value
    track_history=True        # Enable phi history tracking
)
```

## API Reference

### EnhancedVWPRWrapper

#### Methods

- `forward(x)`: Standard forward pass through the wrapped model
- `apply_vwpr_step(base_loss, optimizer_theta, optimizer_phi)`: Performs dual optimization step
- `phi_params()`: Returns iterator over phi parameters for optimizer setup
- `visualize_phi_distribution(epoch)`: Generates phi distribution plots
- `get_phi_history()`: Returns stored phi evolution data

#### Parameters

- `base_model`: The PyTorch model to wrap
- `normalize_phi`: Whether to normalize phi values (default: False)
- `l1_lambda`: L1 regularization coefficient for phi (default: 0.01)
- `phi_clip_value`: Maximum allowed phi value (default: None)
- `track_history`: Enable phi history tracking (default: False)

## Applications

- **Anomaly Detection**: Amplifies learning on rare or unusual patterns
- **Robust Training**: Maintains sensitivity to edge cases and outliers
- **Creative AI**: Preserves unconventional outputs that might be valuable
- **Transfer Learning**: Identifies which learned features are most informative

## Technical Details

### Dual Optimization Process

1. Compute base loss on current batch
2. Calculate local error signals (l_i) for each parameter
3. Update value weights (φ_i) based on error magnitude
4. Apply value-weighted gradients to model parameters
5. Regularize phi values to prevent instability

### Mathematical Foundation

- **Value Weight Update**: `φ_i = softplus(φ_i + α * l_i)`
- **Weighted Loss**: `L_transformed = Σ(φ_i * l_i)`
- **Regularization**: `L_total = L_transformed + λ * ||φ||_1`

## Visualization

The wrapper includes built-in visualization tools to monitor phi weight distributions:

```python
# Generate phi distribution histogram
vwpr_model.visualize_phi_distribution(epoch_number)

# Access raw phi history data
phi_history = vwpr_model.get_phi_history()
```

## Stability Considerations

- Phi values are clipped or regularized to prevent numerical instability
- L1 regularization prevents excessive phi growth
- Optional normalization maintains balanced value distribution
- Softplus activation ensures positive phi values

## Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Matplotlib (for visualization)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{enhanced_vwpr_wrapper,
  title={Enhanced Value-Weighted Parameter Refinement for Neural Networks},
  author={[Your Name]},
  year={2024},
  url={https://github.com/yourusername/enhanced-vwpr-wrapper}
}
```
