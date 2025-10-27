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
    output_text = (f"Ordinal Output Layer\n($k-1 = {k-1}$ Sigmoid Neurons)\n\n"
                   f"$P(y > C_1)$\n"
                   f"$P(y > C_2)$\n"
                   f"...\n"
                   f"$P(y > C_{k-1})$")
    ax.text(0.5, 0.75, output_text, ha='center', va='center', fontsize=12, bbox=output_style_a)

    # --- Draw arrows ---
    # Arrow: Input -> Backbone
    arrow1 = patches.FancyArrowPatch((0.5, 0.22), (0.5, 0.33), **arrowstyle)
    ax.add_patch(arrow1)
    
    # Arrow: Backbone -> Output Layer
    arrow2 = patches.FancyArrowPatch((0.5, 0.47), (0.5, 0.60), **arrowstyle)
    ax.add_patch(arrow2)
    
    # Arrow: Output -> Final Prediction
    ax.text(0.5, 0.95, "Final Class Probabilities\n(via combination)", ha='center', va='center', fontsize=10)
    arrow3 = patches.FancyArrowPatch((0.5, 0.90), (0.5, 0.92), **arrowstyle)
    ax.add_patch(arrow3)

    # ==================================================================
    # Plot (b): Ordinal Loss Function Strategy
    # ==================================================================
    ax = axes[1]
    ax.set_title('(b) Ordinal Loss Function (e.g., CORAL, EMD)', fontsize=14)
    ax.axis('off')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    # --- Draw boxes ---
    ax.text(0.5, 0.15, "Input (Image, Text, etc.)", ha='center', va='center', fontsize=12, bbox=boxstyle)
    ax.text(0.5, 0.4, "CNN / Transformer Backbone", ha='center', va='center', fontsize=12, bbox=boxstyle)
    
    # --- Standard Output Layer (k neurons) ---
    output_text_b = (f"Standard Softmax Layer\n($k = {k}$ Neurons)\n\n"
                     f"$P(y = C_1)$\n"
                     f"$P(y = C_2)$\n"
                     f"...\n"
                     f"$P(y = C_k)$")
    ax.text(0.5, 0.7, output_text_b, ha='center', va='center', fontsize=12, bbox=output_style_b)

    # --- Loss Function Box ---
    ax.text(0.5, 0.92, "Ordinal Loss Function\n(e.g., CORAL, CORN, EMD)", ha='center', va='center', fontsize=12, bbox=loss_style)

    # --- Draw arrows ---
    # Arrow: Input -> Backbone
    arrow1_b = patches.FancyArrowPatch((0.5, 0.22), (0.5, 0.33), **arrowstyle)
    ax.add_patch(arrow1_b)
    
    # Arrow: Backbone -> Output Layer
    arrow2_b = patches.FancyArrowPatch((0.5, 0.47), (0.5, 0.59), **arrowstyle)
    ax.add_patch(arrow2_b)
    
    # Arrow: Output Layer -> Loss Function
    arrow3_b = patches.FancyArrowPatch((0.5, 0.81), (0.5, 0.87), **arrowstyle)
    ax.add_patch(arrow3_b)

    # --- Finalize ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("../../figures/deep_learning_ordinal_models.png", dpi=300)
    plt.show()

# --- Generate the plot ---
# Using k=5 as a common example (e.g., 5-star ratings or 0-4 severity)
plot_deep_learning_models(k=5)
