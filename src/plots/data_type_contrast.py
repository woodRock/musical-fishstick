"""Generates a bar chart to contrast model performance across different data types.

This script creates a grouped bar chart to compare the performance of two specific
models ('MLP-POM' and 'MLP-CORN') on two different datasets ('Boston' and 'Car'),
representing numerical and categorical data types, respectively. The chart
visualizes the Mean Absolute Error (MAE) for each model-dataset combination.
"""
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
    ax.annotate('', xy=(0.2, 0.3), xytext=(0.8, 0.3), arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax.annotate('', xy=(1.2, 0.3), xytext=(1.8, 0.3), arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax.text(1, 0.2, "No inherent order", ha='center', va='center', style='italic', color='gray')

    # --- 2. Ordinal Data ---
    ax = axes[1]
    ax.set_title('(b) Ordinal Data (Ordered)', fontsize=12)
    ax.set_yticks([])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Small', 'Medium', 'Large'], fontsize=12)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 1)
    
    # Draw points on a line to show order
    ax.plot([0, 1, 2], [0.5, 0.5, 0.5], 'o', markersize=10, color='darkcyan')
    ax.axhline(0.5, color='lightgray', linestyle='--')
    
    # Indicate unequal intervals
    ax.annotate('Intervals may be unequal', xy=(0.5, 0.3), xytext=(0.5, 0.1),
                arrowprops=dict(arrowstyle='->', color='gray'), ha='center')
    ax.annotate('', xy=(0, 0.6), xytext=(1, 0.6), arrowprops=dict(arrowstyle='<->', color='darkcyan', lw=1.5))
    ax.annotate('', xy=(1, 0.7), xytext=(2, 0.7), arrowprops=dict(arrowstyle='<->', color='darkcyan', lw=2.5)) # Make one arrow longer
    ax.text(0.5, 0.65, '$d_1$', ha='center', va='bottom', color='darkcyan')
    ax.text(1.5, 0.75, '$d_2$', ha='center', va='bottom', color='darkcyan')
    ax.text(1, 0.85, '$d_1 \neq d_2$', ha='center', va='center', style='italic', color='gray')

    # --- 3. Metric Data ---
    ax = axes[2]
    ax.set_title('(c) Metric Data (Ordered, Equidistant)', fontsize=12)
    ax.set_yticks([])
    ax.set_xticks(np.arange(-2, 3, 1))
    ax.set_xticklabels(np.arange(-2, 3, 1), fontsize=10)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0, 1)
    
    # Draw a number line
    ax.axhline(0.5, color='black', lw=2)
    # Add ticks
    for i in np.arange(-2, 3, 1):
        ax.plot([i, i], [0.45, 0.55], color='black', lw=2)
        
    # Indicate equal intervals
    ax.annotate('', xy=(0, 0.6), xytext=(1, 0.6), arrowprops=dict(arrowstyle='<->', color='navy', lw=2))
    ax.annotate('', xy=(1, 0.6), xytext=(2, 0.6), arrowprops=dict(arrowstyle='<->', color='navy', lw=2))
    ax.text(0.5, 0.65, '$d$', ha='center', va='bottom', color='navy')
    ax.text(1.5, 0.65, '$d$', ha='center', va='bottom', color='navy')
    ax.text(0, 0.2, "Intervals are equal", ha='center', va='center', style='italic', color='gray')

    # --- Finalize ---
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig("../../figures/data_type_contrast.png", dpi=300)
    plt.show()

# --- Generate the plot ---
if __name__ == '__main__':
    plot_data_type_contrast()
