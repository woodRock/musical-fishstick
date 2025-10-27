"""Generates a flowchart illustrating the architecture of deep learning ordinal models.

This script uses the graphviz library to create a diagram that shows the
relationship and structure of various deep learning models for ordinal regression,
such as POM, CORN, and CORAL. The output is saved as a PNG image file.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_deep_learning_models(k=5, figsize=(14, 7)):
    """
    Generates a conceptual plot for Deep Learning Ordinal Models.
    
    Args:
        k (int): Number of ordinal classes for the example (e.g., 5-star rating).
        figsize (tuple): Figure size.
    """
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Family 3: Deep Learning Strategies for Ordinal Classification', fontsize=18, y=0.98)
    
    # --- Define styles ---
    boxstyle = dict(boxstyle='round,pad=0.7', fc='aliceblue', ec='navy', lw=1.5)
    arrowstyle = dict(arrowstyle='-|>', mutation_scale=20, lw=2.0, color='dimgray')
    output_style_a = dict(boxstyle='round,pad=0.7', fc='moccasin', ec='darkorange', lw=1.5)
    output_style_b = dict(boxstyle='round,pad=0.7', fc='honeydew', ec='darkgreen', lw=1.5)
    loss_style = dict(boxstyle='round,pad=0.7', fc='mistyrose', ec='darkred', lw=1.5)

    # ==================================================================
    # Plot (a): Ordinal Output Layer Strategy
    # ==================================================================
    ax = axes[0]
    ax.set_title('(a) Ordinal Output Layer (e.g., Niu et al. 2016)', fontsize=14)
    ax.axis('off')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    # --- Draw boxes ---
    ax.text(0.5, 0.15, "Input (Image, Text, etc.)", ha='center', va='center', fontsize=12, bbox=boxstyle)
    ax.text(0.5, 0.4, "CNN / Transformer Backbone", ha='center', va='center', fontsize=12, bbox=boxstyle)
    
    # --- Output Layer (k-1 neurons) ---
    output_text = (f"Ordinal Output Layer\n({k-1} neurons, sigmoid)")
    ax.text(0.5, 0.65, output_text, ha='center', va='center', fontsize=12, bbox=output_style_a)
    
    # --- Final Prediction ---
    ax.text(0.5, 0.85, f"Predicted Rank = $1 + \\sum_{{i=1}}^{{k-1}} y_i$", ha='center', va='center', fontsize=14)

    # --- Draw arrows ---
    ax.annotate("", xy=(0.5, 0.3), xytext=(0.5, 0.25), arrowprops=arrowstyle)
    ax.annotate("", xy=(0.5, 0.52), xytext=(0.5, 0.48), arrowprops=arrowstyle)
    ax.annotate("", xy=(0.5, 0.75), xytext=(0.5, 0.7), arrowprops=arrowstyle)

    # ==================================================================
    # Plot (b): Ordinal Loss Function Strategy
    # ==================================================================
    ax = axes[1]
    ax.set_title('(b) Ordinal Loss Function (e.g., CORAL, CORN, EMD)', fontsize=14)
    ax.axis('off')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    # --- Draw boxes ---
    ax.text(0.5, 0.15, "Input (Image, Text, etc.)", ha='center', va='center', fontsize=12, bbox=boxstyle)
    ax.text(0.5, 0.4, "CNN / Transformer Backbone", ha='center', va='center', fontsize=12, bbox=boxstyle)
    
    # --- Output Layer (can be varied) ---
    output_text = (f"Standard Output Layer\n(e.g., {k} classes for softmax, or 1 for regression)")
    ax.text(0.5, 0.65, output_text, ha='center', va='center', fontsize=12, bbox=output_style_b)
    
    # --- Ordinal Loss ---
    ax.text(0.5, 0.85, "Ordinal Loss Function\n(e.g., CORAL, CORN)", ha='center', va='center', fontsize=12, bbox=loss_style)

    # --- Draw arrows ---
    ax.annotate("", xy=(0.5, 0.3), xytext=(0.5, 0.25), arrowprops=arrowstyle)
    ax.annotate("", xy=(0.5, 0.52), xytext=(0.5, 0.48), arrowprops=arrowstyle)
    ax.annotate("", xy=(0.5, 0.75), xytext=(0.5, 0.7), arrowprops=arrowstyle)

    # --- Finalize ---
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig("../../figures/deep_learning_ordinal_models.png", dpi=300)
    plt.show()

# --- Generate the plot ---
if __name__ == '__main__':
    plot_deep_learning_models(k=5)
