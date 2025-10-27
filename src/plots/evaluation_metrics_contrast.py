"""Module for generating a conceptual plot to contrast standard and ordinal-aware evaluation metrics."""

import matplotlib.pyplot as plt
import numpy as np

def plot_evaluation_metrics_contrast(k=5, figsize=(12, 6)):
    """
    Generates a conceptual plot contrasting Standard Accuracy with
    Ordinal-aware metrics (like MAE or QWK).

    This function visualizes the difference in how standard classification accuracy
    (which treats all misclassifications equally) and ordinal-aware metrics
    (which penalize misclassifications based on their distance from the true label)
    assign error penalties. It uses heatmaps to represent the weight matrices for
    a given number of classes `k`.

    The plot is saved as 'figures/evaluation_metrics_contrast.png'.

    Args:
        k (int, optional): The number of classes for the example. Defaults to 5.
        figsize (tuple, optional): A tuple (width, height) in inches for the figure size.
                                   Defaults to (12, 6).
    """
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Conceptual Difference in Evaluation Metrics', fontsize=16, y=1.02)
    
    # --- Create the weight matrices ---
    
    # 1. Standard Accuracy (0-1 Loss)
    # Penalizes all incorrect predictions equally.
    accuracy_weights = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                accuracy_weights[i, j] = 1 # All errors have a weight of 1
            else:
                accuracy_weights[i, j] = 0 # Correct predictions have 0 weight
    
    # 2. Ordinal Metric (e.g., MAE/QWK)
    # Penalizes predictions based on distance from the true label.
    # QWK uses squared distance: (i-j)^2
    # MAE uses absolute distance: |i-j|
    ordinal_weights = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            ordinal_weights[i, j] = (i - j)**2 # Using squared error (like QWK)
            
    
    # --- Ticks and Labels ---
    labels = [f'$C_{i+1}
 for i in range(k)]
    ticks = np.arange(k)

    # ==================================================================
    # Plot (a): Standard Classification Accuracy (Nominal)
    # ==================================================================
    ax = axes[0]
    ax.set_title('(a) Standard Accuracy (Nominal Loss)', fontsize=14)
    
    # Use imshow to create the heatmap of weights
    # We use a custom colormap: 0=Green, 1=Red
    cmap_acc = plt.cm.colors.ListedColormap(['mediumseagreen', 'salmon'])
    im_acc = ax.imshow(accuracy_weights, cmap=cmap_acc, vmin=0, vmax=1)
    
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('True Class', fontsize=12)
    
    # Add text annotations
    for i in range(k):
        for j in range(k):
            text_val = "Correct" if i == j else "Incorrect"
            text_color = "black" if i == j else "black"
            ax.text(j, i, text_val, ha='center', va='center', color=text_color, fontsize=9)
            
    ax.text(k/2 - 0.5, -1.0, "All errors are penalized equally.", 
            ha='center', fontsize=12, style='italic')


    # ==================================================================
    # Plot (b): Ordinal Metrics (e.g., QWK / MAE)
    # ==================================================================
    ax = axes[1]
    ax.set_title('(b) Ordinal Metric (e.g., QWK)', fontsize=14)
    
    # Use imshow with a sequential colormap
    cmap_ord = plt.cm.Reds
    cmap_ord.set_under('mediumseagreen') # Set value 0 to green
    
    im_ord = ax.imshow(ordinal_weights, cmap=cmap_ord, vmin=0.1) # vmin > 0 so 0 maps to 'under' color
    
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('True Class', fontsize=12)

    # Add text annotations (the error weight)
    for i in range(k):
        for j in range(k):
            text_val = f"Error = {ordinal_weights[i, j]:.0f}"
            if i == j:
                text_val = "Correct\n(Error=0)"
            text_color = "black"
            if ordinal_weights[i, j] > (k**2 / 2):
                text_color = "white" # Use white text for dark red cells
                
            ax.text(j, i, text_val, ha='center', va='center', color=text_color, fontsize=9)

    ax.text(k/2 - 0.5, -1.0, "Errors are penalized by their distance.", 
            ha='center', fontsize=12, style='italic')

    # Add a colorbar to explain the error gradient
    cbar = fig.colorbar(im_ord, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Error Penalty (Cost)', rotation=270, labelpad=15)

    # --- Finalize ---
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("../../figures/evaluation_metrics_contrast.png", dpi=300)
    plt.show()

# --- Generate the plot ---
if __name__ == '__main__':
    plot_evaluation_metrics_contrast(k=5)
