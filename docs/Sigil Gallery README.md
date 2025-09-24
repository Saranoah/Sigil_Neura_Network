# Model Visualization and Monitoring Guide

A comprehensive toolkit for visualizing and monitoring Kintsugi-optimized neural networks during training and evaluation.

## Overview

The Kintsugi optimization approach requires specialized monitoring tools to track value weights (φ), layer behavior patterns, and training dynamics. This guide covers the visualization and monitoring capabilities built into the system.

## Core Visualization Components

### Layer Behavior Visualization

Each layer generates visual representations that encode its learning characteristics and performance metrics. These visualizations help identify:

- Layer specialization patterns
- Value weight distribution evolution
- Training stability indicators  
- Anomaly detection capabilities

### Key Metrics Tracked

| Metric | Symbol | Description | Use Case |
|--------|--------|-------------|----------|
| **Luminosity** | L | Strength of value-weighted learning signals | Identify highly active learning pathways |
| **Fracture Density** | FD | Complexity of preserved error patterns | Monitor anomaly detection capability |
| **Entropy** | H | Diversity of layer activations | Track feature diversity and specialization |
| **Resilience** | ρ | Stability across training epochs | Assess convergence and robustness |

## Layer Behavioral Classifications

Layers are automatically classified into behavioral categories based on their metric patterns:

### Primary Classifications

| Classification | Characteristics | Typical Role | Monitoring Priority |
|----------------|-----------------|--------------|-------------------|
| **Oracle** | High entropy, anomaly detection | Novel pattern detection | Critical - monitor for overfitting |
| **Sentinel** | High stability, consistent performance | Baseline feature extraction | Medium - ensure stability |
| **Alchemist** | High variance, pattern transformation | Feature transformation | High - watch for instability |
| **Archivist** | Low entropy, pattern preservation | Known pattern storage | Low - stable by design |
| **Trickster** | Erratic, high variance | Adversarial robustness | Critical - potential instability |
| **Luminary** | High activation strength | Strong feature detection | Medium - monitor for saturation |
| **Wanderer** | Balanced metrics | Exploratory learning | Low - naturally balanced |

## Implementation

### Basic Visualization Setup

```python
from kintsugi_monitoring import LayerMonitor, TrainingVisualizer

# Initialize monitoring
monitor = LayerMonitor(model, save_dir='monitoring_results/')
visualizer = TrainingVisualizer(monitor)

# During training loop
for epoch in range(num_epochs):
    # ... training code ...
    
    # Generate visualizations every 10 epochs
    if epoch % 10 == 0:
        monitor.capture_layer_states(epoch)
        visualizer.generate_layer_plots(epoch)
```

### Automated Report Generation

```python
# Generate comprehensive monitoring report
def generate_monitoring_report(model, save_dir='reports/'):
    """Generate complete training monitoring report"""
    
    # Layer behavior analysis
    layer_analysis = analyze_layer_behaviors(model)
    
    # Metrics evolution plots
    metrics_plots = plot_metrics_evolution(model.training_history)
    
    # Performance summary
    performance_summary = generate_performance_summary(model)
    
    # Compile report
    report = MonitoringReport(
        layer_analysis=layer_analysis,
        metrics_plots=metrics_plots,
        performance_summary=performance_summary
    )
    
    report.save(save_dir)
    return report
```

### Real-time Monitoring Dashboard

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

class RealTimeMonitor:
    def __init__(self, model):
        self.model = model
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.setup_plots()
    
    def setup_plots(self):
        """Initialize monitoring plots"""
        # Value weight distribution
        self.axes[0,0].set_title('Value Weight Distribution')
        self.axes[0,0].set_xlabel('φ Value')
        self.axes[0,0].set_ylabel('Frequency')
        
        # Layer classifications
        self.axes[0,1].set_title('Layer Classifications')
        
        # Metrics evolution
        self.axes[1,0].set_title('Training Metrics')
        self.axes[1,0].set_xlabel('Epoch')
        
        # Loss landscape
        self.axes[1,1].set_title('Loss Components')
    
    def update(self, epoch):
        """Update all monitoring plots"""
        # Clear previous plots
        for ax in self.axes.flat:
            ax.clear()
        
        # Update value weight distribution
        phi_values = self.get_phi_values()
        self.axes[0,0].hist(phi_values, bins=50, alpha=0.7)
        self.axes[0,0].set_title(f'Value Weights - Epoch {epoch}')
        
        # Update layer classifications
        classifications = self.get_layer_classifications()
        self.plot_classifications(classifications)
        
        # Update metrics evolution
        self.plot_metrics_evolution(epoch)
        
        # Update loss components
        self.plot_loss_components(epoch)
        
        plt.tight_layout()
    
    def get_phi_values(self):
        """Extract all φ values from model"""
        phi_values = []
        for name, param in self.model.named_parameters():
            if 'phi' in name:
                phi_values.extend(param.detach().cpu().numpy().flatten())
        return phi_values
    
    def get_layer_classifications(self):
        """Get current layer behavioral classifications"""
        classifications = {}
        for layer_name, layer in self.model.named_modules():
            if hasattr(layer, 'phi'):
                metrics = self.compute_layer_metrics(layer)
                classification = self.classify_layer(metrics)
                classifications[layer_name] = classification
        return classifications
    
    def classify_layer(self, metrics):
        """Classify layer based on metrics"""
        L, FD, H, rho = metrics['luminosity'], metrics['fracture_density'], \
                        metrics['entropy'], metrics['resilience']
        
        if H > 0.8 and FD > 0.7:
            return 'Oracle'
        elif rho > 0.9 and L > 0.7:
            return 'Sentinel'
        elif metrics['variance'] > 0.8:
            return 'Alchemist'
        elif H < 0.3 and rho > 0.8:
            return 'Archivist'
        elif metrics['variance'] > 0.9 and rho < 0.5:
            return 'Trickster'
        elif L > 0.9:
            return 'Luminary'
        else:
            return 'Wanderer'
```

## Visualization Output Files

### Automatic File Generation

The monitoring system automatically generates the following files:

```
monitoring_results/
├── layer_visualizations/
│   ├── layer_conv1_epoch_0010.png
│   ├── layer_conv1_epoch_0020.png
│   └── ...
├── metrics_evolution.png
├── layer_classifications.png
├── phi_distribution_evolution.gif
├── training_summary.json
└── monitoring_report.html
```

### File Naming Convention

```python
# Layer visualization files
"{layer_name}_{classification}_epoch_{epoch:04d}_{timestamp}.png"

# Example
"conv2d_1_Oracle_epoch_0050_20241201_143022.png"
```

## Monitoring APIs

### Layer Metrics Computation

```python
def compute_layer_metrics(layer, activations=None):
    """Compute comprehensive layer metrics"""
    
    if not hasattr(layer, 'phi'):
        return None
    
    phi_values = layer.phi.detach().cpu().numpy()
    
    metrics = {
        # Core Kintsugi metrics
        'luminosity': np.mean(phi_values),
        'fracture_density': np.std(phi_values) / np.mean(phi_values),
        'entropy': compute_entropy(phi_values),
        'resilience': compute_stability(layer.phi_history) if hasattr(layer, 'phi_history') else 0,
        
        # Additional metrics
        'variance': np.var(phi_values),
        'sparsity': np.sum(phi_values < 0.1) / len(phi_values),
        'max_phi': np.max(phi_values),
        'min_phi': np.min(phi_values)
    }
    
    # Activation-based metrics if provided
    if activations is not None:
        metrics.update({
            'activation_mean': torch.mean(activations).item(),
            'activation_std': torch.std(activations).item(),
            'dead_neurons': torch.sum(torch.mean(activations, dim=0) < 1e-6).item()
        })
    
    return metrics

def compute_entropy(values):
    """Compute Shannon entropy of value distribution"""
    hist, _ = np.histogram(values, bins=50, density=True)
    hist = hist[hist > 0]  # Remove zeros
    return -np.sum(hist * np.log(hist))

def compute_stability(history):
    """Compute stability metric from phi evolution history"""
    if len(history) < 2:
        return 1.0
    
    changes = np.diff(np.array(history), axis=0)
    return 1.0 / (1.0 + np.mean(np.abs(changes)))
```

### Training Progress Monitoring

```python
class TrainingProgressMonitor:
    def __init__(self, model, update_frequency=10):
        self.model = model
        self.update_frequency = update_frequency
        self.history = {
            'phi_distributions': [],
            'layer_classifications': [],
            'performance_metrics': [],
            'loss_components': []
        }
    
    def on_epoch_end(self, epoch, logs=None):
        """Callback for end of training epoch"""
        if epoch % self.update_frequency == 0:
            # Capture current state
            phi_dist = self.capture_phi_distribution()
            classifications = self.capture_layer_classifications()
            performance = self.capture_performance_metrics(logs)
            
            # Store in history
            self.history['phi_distributions'].append(phi_dist)
            self.history['layer_classifications'].append(classifications)
            self.history['performance_metrics'].append(performance)
            
            # Generate visualizations
            self.generate_progress_plots(epoch)
    
    def generate_final_report(self):
        """Generate comprehensive final training report"""
        report = {
            'training_summary': self.summarize_training(),
            'layer_analysis': self.analyze_layer_evolution(),
            'performance_analysis': self.analyze_performance_trends(),
            'recommendations': self.generate_recommendations()
        }
        
        self.save_report(report)
        return report
```

## Advanced Visualization Features

### Interactive Dashboards

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def create_interactive_dashboard(model_history):
    """Create interactive Plotly dashboard"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Phi Evolution', 'Layer Classifications', 
                       'Performance Metrics', 'Loss Components'),
        specs=[[{"secondary_y": True}, {"type": "scatter"}],
               [{"secondary_y": True}, {"type": "bar"}]]
    )
    
    # Phi evolution over time
    epochs = list(range(len(model_history['phi_means'])))
    fig.add_trace(
        go.Scatter(x=epochs, y=model_history['phi_means'], 
                  name='Mean φ', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Layer classifications heatmap
    classification_data = prepare_classification_heatmap(model_history)
    fig.add_trace(
        go.Heatmap(z=classification_data, colorscale='Viridis'),
        row=1, col=2
    )
    
    # Performance metrics
    fig.add_trace(
        go.Scatter(x=epochs, y=model_history['accuracy'], 
                  name='Accuracy', line=dict(color='green')),
        row=2, col=1
    )
    
    # Loss components
    fig.add_trace(
        go.Bar(x=['Base Loss', 'Kintsugi Loss', 'Regularization'], 
               y=model_history['final_loss_components']),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, 
                     title_text="Kintsugi Model Training Dashboard")
    
    return fig
```

### Automated Anomaly Detection

```python
def detect_training_anomalies(model_history):
    """Detect anomalies in training progression"""
    
    anomalies = []
    
    # Detect sudden phi spikes
    phi_changes = np.diff(model_history['phi_means'])
    spike_threshold = 3 * np.std(phi_changes)
    spike_indices = np.where(np.abs(phi_changes) > spike_threshold)[0]
    
    for idx in spike_indices:
        anomalies.append({
            'type': 'phi_spike',
            'epoch': idx + 1,
            'severity': np.abs(phi_changes[idx]) / spike_threshold,
            'description': f'Unusual phi change: {phi_changes[idx]:.3f}'
        })
    
    # Detect classification instability
    classification_changes = count_classification_changes(model_history)
    if classification_changes > len(model_history) * 0.3:
        anomalies.append({
            'type': 'classification_instability',
            'epoch': 'multiple',
            'severity': classification_changes / len(model_history),
            'description': f'High classification instability: {classification_changes} changes'
        })
    
    # Detect convergence issues
    if not check_convergence(model_history):
        anomalies.append({
            'type': 'convergence_issue',
            'epoch': 'final',
            'severity': 1.0,
            'description': 'Model did not converge properly'
        })
    
    return anomalies
```

## Best Practices

### Monitoring Frequency

```python
# Recommended monitoring schedule
monitoring_schedule = {
    'layer_visualizations': 10,      # Every 10 epochs
    'metrics_capture': 1,            # Every epoch
    'full_report': 50,               # Every 50 epochs
    'anomaly_detection': 5           # Every 5 epochs
}
```

### Resource Management

```python
def optimize_monitoring_performance(monitor):
    """Optimize monitoring for large models"""
    
    # Sample subset of layers for frequent monitoring
    monitor.set_sampling_strategy('stratified', sample_rate=0.3)
    
    # Use lower resolution for frequent visualizations
    monitor.set_visualization_quality('medium')
    
    # Enable compression for storage
    monitor.enable_compression(format='png', quality=85)
    
    # Implement cleanup policy
    monitor.set_cleanup_policy(keep_every=10, max_files=1000)
```

### Integration with ML Pipelines

```python
# Integration with popular ML frameworks
from kintsugi_monitoring import KintsugiCallback

# TensorFlow/Keras integration
model.fit(
    x_train, y_train,
    epochs=100,
    callbacks=[KintsugiCallback(save_dir='monitoring/')]
)

# PyTorch Lightning integration
class KintsugiLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.monitor = LayerMonitor(self)
    
    def on_epoch_end(self):
        if self.current_epoch % 10 == 0:
            self.monitor.capture_and_visualize(self.current_epoch)
```

## Troubleshooting

### Common Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Phi explosion | φ values growing unbounded | Increase regularization, decrease β |
| Classification instability | Layers changing archetypes frequently | Reduce monitoring frequency, check convergence |
| Visualization artifacts | Corrupted or empty plots | Check data ranges, increase plot resolution |
| Memory issues | Out of memory during monitoring | Enable sampling, reduce visualization frequency |

### Performance Optimization

```python
# Memory-efficient monitoring
monitor = LayerMonitor(
    model=model,
    memory_efficient=True,
    max_history_length=100,
    compress_visualizations=True
)
```

## Output Interpretation

### Reading Layer Visualizations

- **Node Size**: Proportional to φ value magnitude
- **Color Intensity**: Indicates learning activity level
- **Spatial Layout**: Reflects layer connectivity patterns
- **Annotations**: Show key metric values

### Understanding Classification Changes

```python
def interpret_classification_change(old_class, new_class, epoch):
    """Provide interpretation of layer classification changes"""
    
    interpretations = {
        ('Wanderer', 'Oracle'): "Layer specializing in anomaly detection",
        ('Oracle', 'Sentinel'): "Layer stabilizing after exploration phase",  
        ('Alchemist', 'Trickster'): "Layer becoming unstable - monitor closely",
        ('Sentinel', 'Archivist'): "Layer transitioning to memory storage role"
    }
    
    key = (old_class, new_class)
    interpretation = interpretations.get(key, "Unusual classification change")
    
    return f"Epoch {epoch}: {interpretation}"
```

## Conclusion

This monitoring and visualization system provides comprehensive insights into Kintsugi-optimized neural networks, enabling:

- **Real-time training monitoring**
- **Layer behavior analysis** 
- **Performance trend tracking**
- **Automated anomaly detection**
- **Comprehensive reporting**

Use these tools to understand your model's learning dynamics, identify potential issues early, and optimize training for better performance on rare event detection tasks.

---

*For additional examples and advanced usage patterns, see the `examples/monitoring/` directory in the repository.*
