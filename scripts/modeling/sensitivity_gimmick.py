import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Set random seed for reproducibility
np.random.seed(42)

plt.rcParams.update({"font.size": 24})
def create_fancy_curves(num_curves, shift_range, steepness=1, noise_level=0.08):
    x = np.linspace(0, 10, 100)
    curves = []
    for shift in np.linspace(-shift_range, shift_range, num_curves):
        # Create a more complex curve using a combination of functions
        base = (1 / (1 + np.exp(-steepness * (x - 5 - shift)))) # sigmoid base
        # Add some waviness
        waviness = 0.1 * np.sin(x + shift)
        # Add some exponential character at the start
        early_curve = 0.2 * (1 - np.exp(-(x + shift)/2))
        
        # Combine the components
        y = base + waviness + early_curve
        
        # Add random noise
        noise = np.random.normal(0, noise_level, len(x))
        y = y + noise
        
        # Normalize to [0,1] range
        y = (y - y.min()) / (y.max() - y.min())
        # Clip values to ensure they stay between 0 and 1
        y = np.clip(y, 0, 1)
        curves.append(y)
    return x, curves

# Create separate figures instead of subplots
def plot_single_figure(x, curves, color, title, fig_number):
    plt.figure(figsize=(8, 6))
    for curve in curves:
        plt.scatter(x, curve, color=color, alpha=0.5, s=10)  # smaller points
    plt.title(title)
    plt.xlabel(f'Parameter')
    plt.ylabel('Outcome')
    plt.tight_layout()
    # get rid of the right and top spines
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # no x and y ticks
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'sensitivity_analysis_{fig_number}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Plot 1: High variance in output (orange curves)
x, curves = create_fancy_curves(5, 2, steepness=2, noise_level=0.08)
plot_single_figure(x, curves, 'orange', 'Influence of first parameter $(P_1)$', 1)

# Plot 2: Low variance in output (purple curves)
x, curves = create_fancy_curves(5, 0.5, steepness=0.5, noise_level=1)
plot_single_figure(x, curves, 'purple', 'Influence of second parameter $(P_2)$', 2)

# Plot 3: Low variance in output (purple curves)
x, curves = create_fancy_curves(5, 0.5, steepness=0.5, noise_level=1)
plot_single_figure(x, curves, 'purple', 'Influence of third parameter $(P_3)$', 3)

# Plot 4: High variance in output (orange curves)
x, curves = create_fancy_curves(5, 2, steepness=2, noise_level=0.08)
plot_single_figure(x, curves, 'orange', 'Influence of second and third\nparameters $(P_2, P_3)$', 4)

