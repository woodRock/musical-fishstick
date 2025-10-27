"""Module for generating conceptual plots of threshold-based ordinal models."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, logistic

def plot_threshold_model(dist_type='logistic', num_classes=5, figsize=(10, 5)):
    """
    Generates a conceptual plot for Threshold-Based Ordinal Models.

    This function visualizes how a continuous latent variable is divided into
    discrete ordinal classes using a set of ordered thresholds (cut-points).
    It supports either a normal or logistic distribution for the latent variable,
    illustrating concepts behind models like Proportional Odds Model (logistic)
    or Ordered Probit Model (normal).

    The plot is saved as 'figures/threshold_based_ordinal_models.png'.

    Args:
        dist_type (str, optional): The type of underlying distribution for the latent variable.
                                   Can be 'normal' or 'logistic'. Defaults to 'logistic'.
        num_classes (int, optional): The number of ordinal classes (k) to display.
                                     Must be at least 2. Defaults to 5.
        figsize (tuple, optional): A tuple (width, height) in inches for the figure size.
                                   Defaults to (10, 5).

    Raises:
        ValueError: If `num_classes` is less than 2 or `dist_type` is not 'normal' or 'logistic'.
    """
    if num_classes < 2:
        raise ValueError("Number of classes must be at least 2.")

    fig, ax = plt.subplots(figsize=figsize)

    # 1. Define the continuous latent variable range
    z = np.linspace(-4, 4, 500)

    # 2. Plot the probability density function (PDF) of the latent variable
    if dist_type == 'normal':
        pdf = norm.pdf(z)
        dist_name = "Normal Distribution"
    elif dist_type == 'logistic':
        # Logistic PDF is 1 / (exp(-z) + 2 + exp(z))
        pdf = np.exp(-z) / (1 + np.exp(-z))**2
        dist_name = "Logistic Distribution"
    else:
        raise ValueError("dist_type must be 'normal' or 'logistic'")

    ax.plot(z, pdf, color='darkblue', linewidth=2, label=f'PDF of Latent Variable $z^*$ ({dist_name})')
    ax.fill_between(z, 0, pdf, color='lightblue', alpha=0.3)

    # 3. Define the k-1 ordered thresholds (cut-points)
    # Distribute them somewhat evenly for visualization
    thresholds = np.linspace(-2, 2, num_classes - 1)
    thresholds_labels = [r'\theta_' + str(i+1) + '

 for i in range(num_classes - 1)]

    # 4. Plot the thresholds as vertical dashed lines
    for i, threshold in enumerate(thresholds):
        ax.axvline(x=threshold, color='darkred', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(threshold + 0.1, ax.get_ylim()[1] * 0.85, thresholds_labels[i],
                color='darkred', fontsize=12, ha='left')

    # 5. Add labels for the ordinal categories
    # Calculate midpoints for category labels
    category_positions = [-3.5] + list(thresholds) + [3.5] # Extend to edges
    category_midpoints = [(category_positions[i] + category_positions[i+1]) / 2 for i in range(len(category_positions) - 1)]

    for i in range(num_classes):
        label_text = f'$C_{i+1}


        ax.text(category_midpoints[i], ax.get_ylim()[1] * 0.1, label_text,
                color='darkgreen', fontsize=14, ha='center', fontweight='bold')
        # Add a subtle shaded region for each category
        if i == 0:
            ax.fill_between(z[z <= thresholds[0]], 0, pdf[z <= thresholds[0]], color='lightgreen', alpha=0.2)
        elif i == num_classes - 1:
            ax.fill_between(z[z >= thresholds[-1]], 0, pdf[z >= thresholds[-1]], color='lightgreen', alpha=0.2)
        else:
            ax.fill_between(z[(z >= thresholds[i-1]) & (z <= thresholds[i])],
                            0, pdf[(z >= thresholds[i-1]) & (z <= thresholds[i])],
                            color='lightgreen', alpha=0.2)


    # Annotations and Aesthetics
    ax.set_title(f'Threshold-Based Ordinal Model ({num_classes} Classes)', fontsize=16)
    ax.set_xlabel('Continuous Latent Variable $z^*

, fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_yticks([]) # Hide y-axis ticks for cleaner look
    ax.set_xlim(-4, 4)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("../../figures/threshold_based_ordinal_models.png", dpi=300)
    plt.show()

# --- Generate the plots ---
if __name__ == '__main__':
    print("Generating figure for Proportional Odds Model (Logistic distribution)...")
    plot_threshold_model(dist_type='logistic', num_classes=5)

    # print("\nGenerating figure for Ordered Probit Model (Normal distribution)...")
    # plot_threshold_model(dist_type='normal', num_classes=4)

