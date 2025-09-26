# Sigil Neural Network

**Value-Weighted Pathway Reinforcement (VWPR) for Adaptive Neural Architectures**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Sigil Network implements a novel neural optimization approach inspired by the Japanese art of Kintsugi - where cracks are repaired with gold, making the restored object more beautiful than the original. Rather than minimizing all errors, this architecture identifies high-value error signals and amplifies the pathways that process them, creating networks that excel at rare event detection and anomaly recognition.

## Key Innovation

Traditional neural networks minimize average error across all samples. Sigil Network introduces **dual-stream optimization** with two parameter types:

- **θ (theta)**: Standard network weights optimized via gradient descent
- **φ (phi)**: Value weights that increase proportionally to local error signals

High-error pathways receive increased φ values, amplifying their learning signals and creating specialized circuits for rare but important patterns.

## Quick Start

```bash
pip install sigil-network
```

```python
import torch
import torch.nn as nn
from sigil import SigilLayer, SigilNetworkManager

# Wrap existing PyTorch layers
model = nn.Sequential(
    SigilLayer(nn.Linear(128, 64)),
    SigilLayer(nn.ReLU()),
    SigilLayer(nn.Linear(64, 32)),
    SigilLayer(nn.Linear(32, 10))
)

# Initialize manager for training and visualization
manager = SigilNetworkManager(model, gallery_dir="results")

# Training loop
for epoch in range(100):
    # Your standard training code here
    # loss = train_step(model, batch)
    
    # Update phi values based on layer performance
    for layer in manager.sigil_layers:
        layer.update_phi()  # Implement your VWPR logic
    
    manager.record_epoch()
    
    # Generate visualizations every 10 epochs
    if epoch % 10 == 0:
        manager.generate_gallery(epoch)
```

## Architecture

The Sigil Network consists of three core components:

### SigilLayer
```python
class SigilLayer(nn.Module):
    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer
        self.phi = nn.Parameter(torch.ones(out_features))  # Value weights
        self.phi_history = []  # Track evolution over time
```

Wraps any PyTorch layer and adds φ parameters for value-weighted pathway reinforcement.

### SigilNetworkManager
Orchestrates training, visualization, and analysis:
- Epoch tracking and φ history management
- Automatic archetype classification based on layer metrics
- Sigil generation and gallery creation
- Network topology visualization

### Visualization System
Generates interpretable "sigils" - visual representations of each layer's learning state:
- **Luminosity**: Average φ values (brightness of reinforcement)
- **Fracture Density**: φ variance (complexity of value distribution)  
- **Entropy**: Activation diversity (information content)
- **Resilience**: Temporal stability across epochs

## Applications

**Anomaly Detection**: Excels at identifying rare events in imbalanced datasets
- Fraud detection in financial transactions
- Medical diagnosis for rare conditions  
- Cybersecurity threat identification

**Creative AI**: Preserves and amplifies unique stylistic patterns
- Art generation with distinctive features
- Music composition with novel harmonies
- Text generation with creative deviations

**Robust Learning**: Prevents catastrophic forgetting of minority classes
- Multi-task learning with varied importance
- Continual learning scenarios
- Transfer learning for specialized domains

## Algorithm Details

The core VWPR update rules implement dual-stream optimization:

**Value Weight Update (φ):**
```
Δφᵢ = β * (E[lᵢ] - λ * φᵢ)
```

**Network Weight Update (θ):**
```
Δθᵢ = -α * ∇θᵢ(L_transformed + Ω(θ))
```

Where:
- `lᵢ`: Local error signal for pathway i
- `β`: Value weight learning rate  
- `λ`: Regularization coefficient
- `α`: Network weight learning rate
- `Ω(θ)`: Weight regularization term

For detailed mathematical formulation, see [docs/ALGORITHM.md](docs/ALGORITHM.md).

## Performance Characteristics

**Strengths:**
- Superior performance on rare event detection
- Robust to class imbalance
- Interpretable learning dynamics through sigil visualization
- Maintains stable performance on majority classes

**Considerations:**
- Increased memory overhead (2x parameters due to φ weights)
- Requires careful hyperparameter tuning (β, λ)
- Longer training time due to dual-stream updates

## Documentation

- **[Algorithm Specification](docs/ALGORITHM.md)**: Mathematical formulation and theoretical foundation
- **[Stability Analysis](docs/STABILITY.md)**: Engineering solutions for robust training  
- **[API Reference](docs/API.md)**: Complete class and method documentation
- **[Examples](examples/)**: Jupyter notebooks with detailed usage scenarios

## Installation

### From PyPI (Recommended)
```bash
pip install sigil-network
```

### From Source
```bash
## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Saranoah/Sigil_Neural_Network.git
cd Sigil_Neural_Network
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

```

### Requirements
- Python 3.8+
- PyTorch 1.9+
- NumPy 1.21+
- Matplotlib 3.5+
- NetworkX 2.6+
- SciPy 1.7+

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Development setup
- Code style requirements  
- Testing procedures
- Pull request process

## Citation

If you use Sigil Network in your research, please cite:

```bibtex
@software{ali2024sigil,
  author = {Ali, Israa},
  title = {Sigil Neural Network: Value-Weighted Pathway Reinforcement for Adaptive Learning},
  year = {2024},
  url = {https://github.com/Saranoah/Sigil_Neura_Network}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Inspired by the Japanese art of Kintsugi, where broken pottery is repaired with gold lacquer, highlighting rather than hiding the damage. This project applies similar principles to neural network optimization - treating high-error pathways as valuable features rather than flaws to be minimized.

---

**Author**: Israa Ali  
**Email**: israali2019@yahoo.com  
**Institution**: Johns Hopkins University - Applied Generative AI Certificate Program
