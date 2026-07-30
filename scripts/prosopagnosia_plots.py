import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
RED = "#e34948"
GREEN = "#008300"
GRAY_MID = "#f0efec"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]


# --- Chart 1: per-unit face selectivity (diverging) ---
selectivity = np.array([0.02, -0.02, -0.01, 0.01, 0.05, 0.03, -0.02, -0.06])
units = [f"unit {i}" for i in range(len(selectivity))]

fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

y_pos = np.arange(len(selectivity))
bar_w = 0.55
colors = [BLUE if val >= 0 else RED for val in selectivity]
ax.bar(y_pos, selectivity, width=bar_w, color=colors, edgecolor="none", zorder=3)

ax.axhline(0, color=BASELINE, linewidth=1.2, zorder=2)

ax.set_xticks(y_pos)
ax.set_xticklabels(units, color=INK_MUTED, fontsize=9)
ax.set_ylim(-0.08, 0.08)
ax.set_ylabel("selectivity (face − object, normalized)", color=INK_SECONDARY, fontsize=10)
ax.tick_params(axis="y", colors=INK_MUTED, labelsize=9)
for spine in ax.spines.values():
    spine.set_visible(False)

# direct labels on the two discussed units
ax.annotate("most face-selective\n(unit 4)", xy=(4, selectivity[4]),
            xytext=(4, 0.065), ha="center", fontsize=9, color=INK_PRIMARY,
            arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.8))
ax.annotate("least face-selective\n(unit 7, control)", xy=(7, selectivity[7]),
            xytext=(7, -0.075), ha="center", fontsize=9, color=INK_PRIMARY,
            arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.8))

ax.set_title("Per-unit face selectivity in the toy network", color=INK_PRIMARY,
             fontsize=12, loc="left", pad=14)
fig.tight_layout()
fig.savefig("images/prosopagnosia_selectivity.png", dpi=200, facecolor=SURFACE)
plt.close(fig)


# --- Chart 2: accuracy before/after lesioning (grouped bars) ---
conditions = ["before\nlesioning", "after lesioning\nface-selective unit", "after lesioning\ncontrol unit"]
face_acc = [1.00, 0.00, 1.00]
obj_acc = [0.37, 1.00, 0.27]

fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

x = np.arange(len(conditions))
width = 0.32
gap = 0.04
x_face = x - width / 2 - gap / 2
x_obj = x + width / 2 + gap / 2

ax.bar(x_face, face_acc, width=width, color=BLUE, edgecolor="none", zorder=3)
ax.bar(x_obj, obj_acc, width=width, color=GREEN, edgecolor="none", zorder=3)

for xi, val in zip(x_face, face_acc):
    ax.text(xi, val + 0.03, f"{val:.2f}", ha="center", fontsize=9, color=INK_PRIMARY)
for xi, val in zip(x_obj, obj_acc):
    ax.text(xi, val + 0.03, f"{val:.2f}", ha="center", fontsize=9, color=INK_PRIMARY)

ax.axhline(0, color=BASELINE, linewidth=1.2, zorder=2)
ax.set_ylim(0, 1.15)
ax.set_xticks(x)
ax.set_xticklabels(conditions, color=INK_SECONDARY, fontsize=9.5)
ax.set_ylabel("test accuracy", color=INK_SECONDARY, fontsize=10)
ax.tick_params(axis="y", colors=INK_MUTED, labelsize=9)
ax.yaxis.grid(True, color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)

handles = [plt.Rectangle((0, 0), 1, 1, facecolor=BLUE, edgecolor="none"),
           plt.Rectangle((0, 0), 1, 1, facecolor=GREEN, edgecolor="none")]
legend = ax.legend(handles, ["face accuracy", "object accuracy"], loc="upper center",
                    bbox_to_anchor=(0.5, 1.16), ncol=2, frameon=False, fontsize=9.5,
                    labelcolor=INK_SECONDARY)

ax.set_title("Lesioning the face-selective unit costs faces, not objects", color=INK_PRIMARY,
             fontsize=12, loc="left", pad=36)
fig.tight_layout()
fig.savefig("images/prosopagnosia_lesion_accuracy.png", dpi=200, facecolor=SURFACE,
            bbox_extra_artists=(legend,), bbox_inches="tight")
plt.close(fig)

print("saved images/prosopagnosia_selectivity.png and images/prosopagnosia_lesion_accuracy.png")
