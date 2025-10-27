""" Module to generate conceptual plots for Threshold-Based Ordinal Models.

This module creates visualizations that illustrate how continuous latent variables
are segmented into discrete ordinal classes using thresholds (cut-points). It supports
both normal and logistic distributions for the latent variable, demonstrating the
concepts behind models like the Proportional Odds Model (logistic) and Ordered Probit
Model (normal).
"""
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
        dist_name = "Normal"
    elif dist_type == 'logistic':
        pdf = logistic.pdf(z)
        dist_name = "Logistic"
    else:
        raise ValueError("dist_type must be 'normal' or 'logistic'.")

    ax.plot(z, pdf, 'b-', lw=2, label=f'{dist_name} Distribution of Latent Variable $z$')
    ax.fill_between(z, pdf, color='lightblue', alpha=0.5)

    # 3. Define and plot the thresholds (cut-points)
    # These are k-1 thresholds for k classes
    thresholds = np.linspace(-1.5, 1.5, num_classes - 1)
    colors = plt.cm.viridis(np.linspace(0, 1, num_classes))

    # 4. Shade the regions for each class and add labels
    # Region for the first class (-inf to threshold_1)
    z_region = np.linspace(-4, thresholds[0], 100)
    ax.fill_between(z_region, logistic.pdf(z_region), color=colors[0], alpha=0.6)
    ax.text(thresholds[0] - 0.8, 0.1, f'Class 1', ha='center', fontsize=12, weight='bold')

    # Regions for intermediate classes
    for i in range(num_classes - 2):
        z_region = np.linspace(thresholds[i], thresholds[i+1], 100)
        ax.fill_between(z_region, logistic.pdf(z_region), color=colors[i+1], alpha=0.6)
        ax.text((thresholds[i] + thresholds[i+1]) / 2, 0.1, f'Class {i+2}', ha='center', fontsize=12, weight='bold')

    # Region for the last class (threshold_{k-1} to +inf)
    z_region = np.linspace(thresholds[-1], 4, 100)
    ax.fill_between(z_region, logistic.pdf(z_region), color=colors[-1], alpha=0.6)
    ax.text(thresholds[-1] + 0.8, 0.1, f'Class {num_classes}', ha='center', fontsize=12, weight='bold')

    # Plot threshold lines
    for i, t in enumerate(thresholds):
        ax.axvline(t, color='red', linestyle='--', lw=1.5)
        ax.text(t, 0.35, f'$\Theta_{i+1}$', ha='center', fontsize=14, color='red')

    # 5. Final plot formatting
    ax.set_title(f'Family 1: Threshold-Based Ordinal Model (k={num_classes})', fontsize=16)
    ax.set_xlabel(r'Latent Variable $z = X\beta$', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(bottom=0)
    ax.set_xlim(-4, 4)

    plt.tight_layout()
    plt.savefig("../../figures/threshold_based_ordinal_models.png", dpi=300)
    plt.show()

# --- Generate the plot ---
if __name__ == '__main__':
    plot_threshold_model(dist_type='logistic', num_classes=5)