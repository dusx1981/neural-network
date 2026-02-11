# AGENTS.md

Guidelines for AI agents working in this neural network visualization repository.

## Project Overview

This is an educational Python project demonstrating neural network concepts through visualizations. It uses PyTorch for neural network implementations and Matplotlib for visualizations.

## Build/Test/Lint Commands

**No formal build system is configured.**

### Running Code
```bash
# Run a specific visualization script
python code/multi_class_classification.py

# Run with headless mode (no GUI, saves images only)
python code/universal_approximation_demo.py --headless

# Run individual scripts in subdirectories
python code/point_softmax/circular_boundary_visualization.py
python code/point_softmax/gradient_descent_visualization.py
```

### Testing
- **No test suite is configured.**
- Test scripts manually by running them and verifying outputs.
- Check for generated `.png` and `.gif` files in the respective directories.

### Linting (Not Configured)
If adding linting, use these commands:
```bash
# Recommended if setting up linting
flake8 code/
pylint code/
black --check code/
```

## Code Style Guidelines

### Python Style

**Imports** - Group and order as follows:
```python
# 1. Standard library imports
import math
import time

# 2. Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_blobs, make_circles
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

# 3. Local imports (none in this project)
```

**Naming Conventions:**
- Functions: `snake_case` (e.g., `make_data`, `train_model`)
- Classes: `PascalCase` (e.g., `Network`)
- Variables: `snake_case` (e.g., `n_hidden`, `learning_rate`)
- Constants: `UPPER_CASE` (e.g., `COLORS`)
- Private functions: Prefix with underscore `_helper_function`

**Function Documentation:**
Use docstrings for all public functions following this pattern:
```python
def function_name(param1, param2):
    """
    Brief description of what the function does.
    
    Args:
        param1 (type): Description of param1
        param2 (type): Description of param2
    
    Returns:
        type: Description of return value
    """
    pass
```

**Matplotlib Configuration:**
Always set Chinese font support at the top of visualization scripts:
```python
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
```

### Error Handling

Use try-except for optional features like saving animations:
```python
try:
    anim.save('animation.gif', writer='pillow', fps=5)
    print("Animation saved successfully")
except Exception as e:
    print(f"Failed to save animation: {e}")
    # Fallback behavior
```

### Visualization Guidelines

**Figure Sizes:**
- Standard plots: `figsize=(10, 8)` or `figsize=(12, 8)`
- Complex multi-subplots: `figsize=(16, 10)` or `figsize=(20, 14)`
- Animations: `figsize=(16, 10)`

**Color Schemes:**
Use the project's standard color palette:
```python
COLORS = {
    'red': '#FF6B6B',
    'green': '#51CF66', 
    'blue': '#339AF0',
    'yellow': '#FFD93D',
    'purple': '#9C36B5',
    'orange': '#FF922B'
}
```

**Saving Figures:**
Always save before showing:
```python
plt.savefig('output.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()
```

## Project Structure

```
code/
├── multi_class_classification.py          # Multi-class classification demo
├── universal_approximation_demo.py        # Universal approximation theorem demo
└── point_softmax/
    ├── circular_boundary_visualization.py # Circular boundary visualization
    └── gradient_descent_visualization.py  # Gradient descent animation
```

## Dependencies

Key dependencies (install via pip or conda):
- `torch` - Neural network framework
- `numpy` - Numerical computations
- `matplotlib` - Visualization
- `scikit-learn` - Data generation utilities
- `pillow` - For saving GIF animations (optional)

## VS Code Settings

The project uses Conda as the default environment manager (configured in `.vscode/settings.json`).

## Git Workflow

- Commit messages are in Chinese
- No pre-commit hooks configured
- Generated images (.png, .gif) are committed to the repository

## Notes for Agents

1. **No formal testing** - Visual outputs must be verified manually
2. **Chinese comments and output** - Maintain Chinese language for consistency
3. **Educational focus** - Code should be readable and well-commented for learning purposes
4. **Visualization-heavy** - Most scripts generate plots; test with `--headless` flag when available
5. **Random seeds** - Always set `np.random.seed()` for reproducible results
