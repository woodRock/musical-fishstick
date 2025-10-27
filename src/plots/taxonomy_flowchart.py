"""Module for generating a conceptual flowchart illustrating the taxonomy of ordinal classification methods."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def connect(posA, posB, style='->', lw=1):
    """Draw a connection patch between two points on the plot.

    Args:
        posA (tuple): (x, y) coordinates of the starting point.
        posB (tuple): (x, y) coordinates of the ending point.
        style (str, optional): Arrow style for the connection. Defaults to '->'.
        lw (int, optional): Line width of the connection. Defaults to 1.
    """
    # lw (linewidth) must be an argument for ConnectionPatch,
    # not part of the arrowstyle string.
    con = patches.ConnectionPatch(posA, posB, 'data', 'data',
                                  arrowstyle=style, shrinkA=10, shrinkB=10,
                                  ec='gray', lw=lw)
    ax.add_patch(con)

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

# Define box styles
box_props = dict(boxstyle='round,pad=0.5', fc='white', ec='b', lw=1)
main_props = dict(boxstyle='round,pad=0.5', fc='aliceblue', ec='b', lw=2)
family_props = dict(boxstyle='round,pad=0.5', fc='honeydew', ec='g', lw=1.5)
naive_props = dict(boxstyle='round,pad=0.5', fc='ivory', ec='orange', lw=1.5)
model_props = dict(boxstyle='round,pad=0.5', fc='white', ec='gray', lw=1)

# --- Define box positions ---
# Level 0: Main Title
ax.text(6, 9.5, "Taxonomy of Ordinal Classification Methods",
        ha='center', va='center', fontsize=16, weight='bold')

# Level 1: Root
l1_y = 8
ax.text(6, l1_y, "Ordinal Classification", ha='center', va='center',
        bbox=main_props, fontsize=12)

# Level 2: Main Branches
l2_y = 6.5
ax.text(3, l2_y, "Naive Approaches", ha='center', va='center',
        bbox=naive_props, fontsize=11)
ax.text(9, l2_y, "Specialized Ordinal Models", ha='center', va='center',
        bbox=main_props, fontsize=11)

# Level 3: Naive Methods
l3_y_naive = 5
ax.text(1.5, l3_y_naive, "Nominal Classification\n(Ignore Order)",
        ha='center', va='center', bbox=model_props)
ax.text(4.5, l3_y_naive, "Metric Regression\n(Assume Equidistance)",
        ha='center', va='center', bbox=model_props)

# Level 3: Specialized Families
l3_y_spec = 5
ax.text(6, l3_y_spec, "Family 1:\nThreshold-Based", ha='center',
        va='center', bbox=family_props)
ax.text(9, l3_y_spec, "Family 2:\nBinary Decomposition", ha='center',
        va='center', bbox=family_props)
ax.text(11.2, l3_y_spec, "Family 3:\nDeep Learning", ha='center',
         va='center', bbox=family_props) # Moved for better spacing

# Level 4: Specific Models
l4_y = 3.5
ax.text(6, l4_y, "POM, CLMs, SVOR", ha='center',
        va='center', bbox=model_props)
ax.text(9, l4_y, "Cumulative, Adjacent", ha='center',
        va='center', bbox=model_props)
ax.text(11.2, l4_y, "Ordinal Loss (CORAL, EMD)\nOrdinal Output Layer", ha='center',
         va='center', bbox=model_props)

# --- Draw Connecting Lines ---
# L1 to L2
connect((6, l1_y), (3, l2_y))
connect((6, l1_y), (9, l2_y))

# L2 Naive to L3
connect((3, l2_y), (1.5, l3_y_naive))
connect((3, l2_y), (4.5, l3_y_naive))

# L2 Specialized to L3
connect((9, l2_y), (6, l3_y_spec))
connect((9, l2_y), (9, l3_y_spec))
connect((9, l2_y), (11.2, l3_y_spec))

# L3 Families to L4
connect((6, l3_y_spec), (6, l4_y))
connect((9, l3_y_spec), (9, l4_y))
connect((11.2, l3_y_spec), (11.2, l4_y))

# Save and show
plt.tight_layout()
plt.savefig("../../figures/taxonomy_flowchart.png", dpi=300)
print("Taxonomy flowchart saved as 'figures/taxonomy_flowchart.png'")
plt.show()
