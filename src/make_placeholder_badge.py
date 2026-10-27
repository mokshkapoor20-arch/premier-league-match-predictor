import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def make_placeholder(filename, color, initials):
    os.makedirs("static", exist_ok=True)
    fig, ax = plt.subplots(figsize=(1.6, 1.6), dpi=100)
    ax.add_patch(plt.Circle((0.5, 0.5), 0.48, color=color))
    ax.text(0.5, 0.5, initials, ha="center", va="center",
            fontsize=30, fontweight="bold", color="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(f"static/{filename}", transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

# Filename must match the app's pattern: spaces -> "_", apostrophes removed
make_placeholder("Nottm_Forest.png", "#DD0000", "NF")
print("Saved static/Nottm_Forest.png")