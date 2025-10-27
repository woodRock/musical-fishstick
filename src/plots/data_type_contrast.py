"""Module for generating a conceptual plot to contrast different data types."""

import matplotlib.pyplot as plt
import numpy as np

def plot_data_type_contrast(figsize=(10, 4)):
    """
    Generates a conceptual plot contrasting Nominal, Ordinal, and Metric data.

    This function creates a figure with three subplots, each illustrating the
    characteristics of a different data type:
    - Nominal data: Categories without inherent order (e.g., Apple, Banana, Orange).
    - Ordinal data: Categories with a meaningful order but unequal intervals (e.g., Small, Medium, Large).
    - Metric data: Ordered data with meaningful and equidistant intervals (e.g., a number line).

    The plot is saved as 'figures/data_type_contrast.png'.

    Args:
        figsize (tuple, optional): A tuple (width, height) in inches for the figure size.
                                   Defaults to (10, 4).
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    fig.suptitle('Distinguishing Data Types: Nominal, Ordinal, and Metric', fontsize=16, y=1.05)

    # --- 1. Nominal Data ---
    ax = axes[0]
    ax.set_title('(a) Nominal Data (No Order)', fontsize=12)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 1)

    # Draw categorical items (e.g., shapes or colors)
    # Using text for simplicity, could be replaced with actual patches/images
    ax.text(0, 0.5, "Apple", ha='center', va='center', fontsize=16, bbox=dict(boxstyle="round,pad=0.3", fc="red", ec="black", lw=1))
    ax.text(1, 0.5, "Banana", ha='center', va='center', fontsize=16, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1))
    ax.text(2, 0.5, "Orange", ha='center', va='center', fontsize=16, bbox=dict(boxstyle="round,pad=0.3", fc="orange", ec="black", lw=1))
    
    ax.text(1, 0.15, "Order: None", ha='center', va='center', fontsize=12, color='darkred', fontweight='bold')


    # --- 2. Ordinal Data ---
    ax = axes[1]
    ax.set_title('(b) Ordinal Data (Has Order)', fontsize=12)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 1)

    # Draw ordered items (e.g., sizes: Small, Medium, Large)
    ax.text(0, 0.5, "Small", ha='center', va='center', fontsize=16, bbox=dict(boxstyle="round,pad=0.3", fc="lightgray", ec="black", lw=1))
    ax.text(1, 0.5, "Medium", ha='center', va='center', fontsize=16, bbox=dict(boxstyle="round,pad=0.3", fc="gray", ec="black", lw=1))
    ax.text(2, 0.5, "Large", ha='center', va='center', fontsize=16, bbox=dict(boxstyle="round,pad=0.3", fc="darkgray", ec="black", lw=1))

    # Add order arrows
    ax.arrow(0.3, 0.65, 0.4, 0, head_width=0.05, head_length=0.1, fc='blue', ec='blue', lw=1.5)
    ax.arrow(1.3, 0.65, 0.4, 0, head_width=0.05, head_length=0.1, fc='blue', ec='blue', lw=1.5)
    
    ax.text(1, 0.15, "Order: Yes (e.g., $S < M < L$)", ha='center', va='center', fontsize=12, color='darkgreen', fontweight='bold')


    # --- 3. Metric (Regression) Data ---
    ax = axes[2]
    ax.set_title('(c) Metric Data (Ordered, Equidistant)', fontsize=12)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)

    # Draw a number line / ruler
    ax.plot([0.5, 4.5], [0.5, 0.5], color='black', lw=2) # Main line
    for i in range(1, 5): # Tick marks
        ax.plot([i, i], [0.45, 0.55], color='black', lw=1.5)
        ax.text(i, 0.35, str(i), ha='center', va='top', fontsize=14, fontweight='bold')
    
    ax.text(2.5, 0.15, "Order: Yes (e.g., $1 \prec 2 \prec 3 \dots$)", ha='center', va='center', fontsize=12, color='purple', fontweight='bold')
    ax.text(2.5, 0.8, "Distance: Measurable & Meaningful\n(e.g., $2-1 = 1$, $3-2=1$)", ha='center', va='bottom', fontsize=10, color='darkblue')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout to make space for suptitle
    plt.savefig("../../figures/data_type_contrast.png", dpi=300)
    plt.show()

# Generate the plot
if __name__ == '__main__':
    plot_data_type_contrast()