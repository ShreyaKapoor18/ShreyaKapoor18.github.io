import numpy as np
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]

np.random.seed(0)


def make_dataset(n=2000, size=16):
    """Synthetic 'faces' (fixed eyes/mouth triplet) vs. 'objects' (randomly placed blob triplets)."""
    X = np.zeros((n, 1, size, size), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        img = np.zeros((size, size), dtype=np.float32)
        if i % 2 == 0:
            for (r, c) in [(4, 5), (4, 10), (10, 7)]:
                img[r - 1:r + 2, c - 1:c + 2] = 1.0
            y[i] = 1
        else:
            for _ in range(3):
                r, c = np.random.randint(2, size - 2, 2)
                img[r - 1:r + 2, c - 1:c + 2] = 1.0
            y[i] = 0
        img += np.random.normal(0, 0.05, img.shape).astype(np.float32)
        X[i, 0] = img
    return X, y


X, y = make_dataset(2000)
faces = X[y == 1][:6]
objects = X[y == 0][:6]

fig, axes = plt.subplots(2, 6, figsize=(9, 3.2), facecolor=SURFACE)

for ax, img in zip(axes[0], faces):
    ax.imshow(img[0], cmap="gray_r", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

for ax, img in zip(axes[1], objects):
    ax.imshow(img[0], cmap="gray_r", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

axes[0, 0].set_ylabel("faces", color=INK_PRIMARY, fontsize=11)
axes[1, 0].set_ylabel("objects", color=INK_PRIMARY, fontsize=11)

fig.suptitle("Sampled synthetic training images", color=INK_PRIMARY, fontsize=12, x=0.09, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("images/prosopagnosia_samples.png", dpi=200, facecolor=SURFACE)
plt.close(fig)

print("saved images/prosopagnosia_samples.png")
