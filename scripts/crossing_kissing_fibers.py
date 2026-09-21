"""Generates the five figures for the 'Crossing and Kissing Fibers' post.

All figures are schematic/synthetic (no external dataset download): the FA
map and classification mask use an analytic two-tensor volume-fraction model
on a hand-built voxel grid, and the fODF plots use dipy's multi_tensor_odf on
a real spherical sampling grid. This mirrors the synthetic-phantom approach
already used elsewhere in the post.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from dipy.data import get_sphere
from dipy.sims.voxel import multi_tensor_odf

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
RED = "#e34948"
GREEN = "#2a9d6b"
ORANGE = "#d98a2b"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]

np.random.seed(0)


def style_axes(ax, title):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left", pad=10)


# ---------------------------------------------------------------------------
# Shared voxel grid: a horizontal "highway" bundle (0 deg) with a crossing
# branch (90 deg) on the left and a near-parallel "kissing" branch (~15 deg)
# on the right.
# ---------------------------------------------------------------------------

n_rows, n_cols = 90, 150
LAM_PAR, LAM_PERP = 1.7e-3, 0.3e-3


def band_a_mask():
    rows = np.arange(n_rows)[:, None]
    return (rows >= 38) & (rows < 53)


def band_b_mask():
    cols = np.arange(n_cols)[None, :]
    return (cols >= 20) & (cols < 35)


def band_c_mask():
    rows = np.arange(n_rows)[:, None]
    cols = np.arange(n_cols)[None, :]
    center = 45 - 0.22 * (cols - 120)
    in_cols = (cols >= 100) & (cols < 150)
    return in_cols & (np.abs(rows - center) < 8)


band_a = np.broadcast_to(band_a_mask(), (n_rows, n_cols))
band_b = np.broadcast_to(band_b_mask(), (n_rows, n_cols))
band_c = band_c_mask()

angle_a = 0.0
angle_b = 90.0
angle_c = 15.0


def tensor_from_angle(angle_deg):
    theta = np.radians(angle_deg)
    d = np.array([np.cos(theta), np.sin(theta)])
    return LAM_PAR * np.outer(d, d) + LAM_PERP * (np.eye(2) - np.outer(d, d))


def fa_from_eigs(eigs):
    eigs = np.clip(eigs, 1e-12, None)
    mean = eigs.mean()
    num = np.sqrt(((eigs - mean) ** 2).sum())
    den = np.sqrt((eigs ** 2).sum())
    return np.sqrt(1.5) * num / den


fa_map = np.zeros((n_rows, n_cols))
principal_dir = np.zeros((n_rows, n_cols))
n_bundles_map = np.zeros((n_rows, n_cols), dtype=int)
angle_sep_map = np.full((n_rows, n_cols), np.nan)

rng = np.random.default_rng(0)

for i in range(n_rows):
    for j in range(n_cols):
        present = []
        if band_a[i, j]:
            present.append(angle_a)
        if band_b[i, j]:
            present.append(angle_b)
        if band_c[i, j]:
            present.append(angle_c)

        n_bundles_map[i, j] = len(present)

        if len(present) == 0:
            fa_map[i, j] = 0.12 + rng.normal(0, 0.02)
            principal_dir[i, j] = rng.uniform(0, 180)
        elif len(present) == 1:
            D = tensor_from_angle(present[0])
            eigs, vecs = np.linalg.eigh(D)
            fa_map[i, j] = fa_from_eigs(eigs) + rng.normal(0, 0.01)
            principal_dir[i, j] = present[0]
        else:
            a, b = present
            angle_sep_map[i, j] = min(abs(a - b), 180 - abs(a - b))
            D = 0.5 * tensor_from_angle(a) + 0.5 * tensor_from_angle(b)
            eigs, vecs = np.linalg.eigh(D)
            fa_map[i, j] = fa_from_eigs(eigs) + rng.normal(0, 0.01)
            principal_vec = vecs[:, np.argmax(eigs)]
            principal_dir[i, j] = np.degrees(np.arctan2(principal_vec[1], principal_vec[0])) % 180

fa_map = np.clip(fa_map, 0, 1)

# ---------------------------------------------------------------------------
# Figure 1: DTI FA map with low-FA crossing regions + principal eigenvector
# ---------------------------------------------------------------------------

fa_cmap = LinearSegmentedColormap.from_list("fa", ["#0b0b0b", "#2a4d8f", "#6fa8e0", "#eef4fb"])

fig, ax = plt.subplots(figsize=(9, 5.4), facecolor=SURFACE)
im = ax.imshow(fa_map, cmap=fa_cmap, vmin=0, vmax=0.9, origin="upper")
step = 6
ys, xs = np.mgrid[0:n_rows:step, 0:n_cols:step]
u = np.cos(np.radians(principal_dir[::step, ::step]))
v = np.sin(np.radians(principal_dir[::step, ::step]))
mag = fa_map[::step, ::step]
mask = mag > 0.2
ax.quiver(xs[mask], ys[mask], u[mask], v[mask], color=INK_PRIMARY,
          scale=28, width=0.003, headwidth=0, headlength=0, alpha=0.85)
ax.set_xlim(0, n_cols)
ax.set_ylim(n_rows, 0)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.annotate("crossing\n(FA collapses,\nvector meaningless)", xy=(27, 45), xytext=(27, 78),
            color=INK_PRIMARY, fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=1))
ax.annotate("single fiber\n(FA high)", xy=(75, 45), xytext=(75, 12),
            color=INK_PRIMARY, fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=1))
ax.annotate("kissing\n(still ~single direction)", xy=(122, 45), xytext=(122, 78),
            color=INK_PRIMARY, fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=1))
ax.set_title("DTI FA map with principal eigenvector — crossing regions collapse FA",
             color=INK_PRIMARY, fontsize=12, loc="left", pad=12)
cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
cbar.set_label("FA", color=INK_SECONDARY, fontsize=9)
fig.tight_layout()
fig.savefig("images/dti_fa_crossings.png", dpi=200, facecolor=SURFACE)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: crossing vs. kissing classification map
# ---------------------------------------------------------------------------

crossing_mask = (n_bundles_map >= 2) & (angle_sep_map > 30)
kissing_mask = (n_bundles_map >= 2) & (angle_sep_map <= 30)
single_mask = n_bundles_map == 1

class_map = np.zeros((n_rows, n_cols, 3))
class_map[...] = np.array([0.98, 0.98, 0.97])  # background
for mask, hexcolor in [(single_mask, "#c9c6bd"), (crossing_mask, BLUE), (kissing_mask, ORANGE)]:
    rgb = np.array([int(hexcolor.lstrip("#")[k:k + 2], 16) / 255 for k in (0, 2, 4)])
    class_map[mask] = rgb

fig, ax = plt.subplots(figsize=(9, 5.4), facecolor=SURFACE)
ax.imshow(class_map, origin="upper")
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
handles = [
    plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=BLUE, markersize=12, label="crossing (sep > 30°)"),
    plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=ORANGE, markersize=12, label="kissing (sep ≤ 30°)"),
    plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="#c9c6bd", markersize=12, label="single fiber"),
]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=3,
          frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
ax.set_title("Crossing vs. kissing classification from CSD peak geometry",
             color=INK_PRIMARY, fontsize=12, loc="left", pad=12)
fig.tight_layout()
fig.savefig("images/crossing_kissing_mask.png", dpi=200, facecolor=SURFACE)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: synthetic crossing phantom geometry
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

rng2 = np.random.default_rng(1)
x = np.linspace(-1.3, 1.3, 200)
for offset in np.linspace(-0.9, 0.9, 9):
    wobble = 0.015 * np.sin(2 * np.pi * x / 1.7 + rng2.uniform(0, 6.3))
    ax.plot(x, offset + wobble, color=BLUE, alpha=0.35, lw=1.2)
y = np.linspace(-1.3, 1.3, 200)
for offset in np.linspace(-0.9, 0.9, 9):
    wobble = 0.015 * np.sin(2 * np.pi * y / 1.7 + rng2.uniform(0, 6.3))
    ax.plot(offset + wobble, y, color=RED, alpha=0.35, lw=1.2)

voxel = plt.Rectangle((-0.5, -0.5), 1.0, 1.0, fill=False, edgecolor=INK_PRIMARY,
                       lw=1.5, linestyle="--", zorder=5)
ax.add_patch(voxel)
ax.annotate("", xy=(0.75, 0), xytext=(-0.75, 0),
            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.5))
ax.annotate("", xy=(0, 0.75), xytext=(0, -0.75),
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.5))
ax.text(0.78, 0.06, "bundle 1, 50%\n(90°, 0°)", color=BLUE, fontsize=9)
ax.text(0.06, 0.78, "bundle 2, 50%\n(90°, 90°)", color=RED, fontsize=9)
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_title("Synthetic crossing phantom: two 50/50 populations at 90°",
             color=INK_PRIMARY, fontsize=12, loc="left", pad=12)
fig.tight_layout()
fig.savefig("images/phantom_crossing_geometry.png", dpi=200, facecolor=SURFACE)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 4: CSD fODF peaks, crossing voxel vs. kissing voxel
# ---------------------------------------------------------------------------

sphere = get_sphere(name="repulsion724")
mevals = np.array([[0.0017, 0.0003, 0.0003], [0.0017, 0.0003, 0.0003]])

equator = np.abs(sphere.vertices[:, 2]) < 0.05
verts_eq = sphere.vertices[equator]
theta_eq = np.arctan2(verts_eq[:, 1], verts_eq[:, 0])
order = np.argsort(theta_eq)
theta_eq = theta_eq[order]

configs = [
    ("Crossing voxel (two peaks, 90° apart)", [(90, 0), (90, 90)], BLUE),
    ("Kissing voxel (peaks 15° apart, effectively one lobe)", [(90, 0), (90, 15)], ORANGE),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), subplot_kw={"projection": "polar"}, facecolor=SURFACE)
for ax, (title, angles, color) in zip(axes, configs):
    odf = multi_tensor_odf(sphere.vertices, mevals, angles=angles, fractions=[50, 50])
    odf_eq = odf[equator][order]
    theta_full = np.concatenate([theta_eq, theta_eq[:1]])
    odf_full = np.concatenate([odf_eq, odf_eq[:1]])
    ax.set_facecolor(SURFACE)
    ax.plot(theta_full, odf_full, color=color, lw=2)
    ax.fill(theta_full, odf_full, color=color, alpha=0.25)
    ax.set_yticklabels([])
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_title(title, color=INK_PRIMARY, fontsize=10, pad=18)

fig.suptitle("fODF cross-section: peak separation distinguishes crossing from kissing",
             color=INK_PRIMARY, fontsize=12, x=0.02, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("images/csd_peaks_crossing_vs_kissing.png", dpi=200, facecolor=SURFACE)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 5: probabilistic tractogram through a crossing region
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5.4), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

box = plt.Rectangle((-1.0, -1.0), 2.0, 2.0, fill=True, facecolor=GRID, alpha=0.4,
                     edgecolor="none", zorder=0)
ax.add_patch(box)

rng3 = np.random.default_rng(2)


def probabilistic_path(start, base_angle, n_steps=110, step=0.035, crossing_box=1.0,
                        max_dev_outside=np.radians(4), max_dev_inside=np.radians(22)):
    pos = np.array(start, dtype=float)
    deviation = 0.0
    pts = [pos.copy()]
    for _ in range(n_steps):
        in_box = abs(pos[0]) < crossing_box and abs(pos[1]) < crossing_box
        max_dev = max_dev_inside if in_box else max_dev_outside
        deviation += rng3.normal(0, np.radians(2.2))
        deviation = np.clip(deviation, -max_dev, max_dev)
        dir_angle = base_angle + deviation
        pos = pos + step * np.array([np.cos(dir_angle), np.sin(dir_angle)])
        pts.append(pos.copy())
        if abs(pos[0]) > 1.9 or abs(pos[1]) > 1.9:
            break
    return np.array(pts)


for y0 in np.linspace(-0.8, 0.8, 9):
    for _ in range(3):
        path = probabilistic_path((-1.9, y0), 0.0)
        ax.plot(path[:, 0], path[:, 1], color=BLUE, alpha=0.4, lw=1.0)

for x0 in np.linspace(-0.8, 0.8, 9):
    for _ in range(3):
        path = probabilistic_path((x0, -1.9), np.pi / 2)
        ax.plot(path[:, 0], path[:, 1], color=RED, alpha=0.4, lw=1.0)

crossing_sq = plt.Rectangle((-1.0, -1.0), 2.0, 2.0, fill=False, edgecolor=INK_PRIMARY,
                             lw=1.5, linestyle="--", zorder=5)
ax.add_patch(crossing_sq)
ax.text(0, 1.55, "probabilistic direction getter samples the full fODF —\n"
                  "streamlines spread down both branches inside the crossing",
        color=INK_PRIMARY, fontsize=9.5, ha="center",
        bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=4))

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
handles = [
    plt.Line2D([0], [0], color=BLUE, lw=2, label="bundle 1 (left–right)"),
    plt.Line2D([0], [0], color=RED, lw=2, label="bundle 2 (up–down)"),
]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2,
          frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
ax.set_title("Probabilistic tractogram through a crossing region",
             color=INK_PRIMARY, fontsize=12, loc="left", pad=12)
fig.tight_layout()
fig.savefig("images/prob_tractogram_crossings.png", dpi=200, facecolor=SURFACE)
plt.close(fig)

print("saved: dti_fa_crossings.png, crossing_kissing_mask.png, phantom_crossing_geometry.png, "
      "csd_peaks_crossing_vs_kissing.png, prob_tractogram_crossings.png")
