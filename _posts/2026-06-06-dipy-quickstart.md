---
title: 'Getting Started with DIPY'
date: 2026-06-06
permalink: /posts/2026-06-06-dipy-quickstart/
tags:
  - Neuroimaging
  - DIPY
  - Diffusion MRI
---

## Getting Started with DIPY

[DIPY](https://dipy.org/) is an open-source Python library for the analysis of diffusion MRI data. In this first tutorial I walk through the basics: loading data, building a gradient table, and visualizing axial slices.

### The three file types in diffusion MRI

Every diffusion MRI acquisition produces three files:

- **NIfTI (`.nii.gz`)** — the 4D diffusion-weighted image volume
- **b-values (`.bval`)** — one scalar per volume encoding the diffusion weighting strength
- **b-vectors (`.bvec`)** — one unit vector per volume encoding the gradient direction

DIPY provides utilities to load all three and combine them into a `GradientTable` that tracks which direction each volume was acquired in.

### Downloading a dataset

DIPY ships with a data fetcher for several public datasets. The Sherbrooke 3-shell dataset is a good starting point:

```python
from dipy.data import fetch_sherbrooke_3shell
from dipy.io import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.core.gradients import gradient_table

fetch_sherbrooke_3shell()

home = Path("~").expanduser()
dname = home / ".dipy" / "sherbrooke_3shell"

data, affine = load_nifti(dname / "HARDI193.nii.gz")
bvals, bvecs = read_bvals_bvecs(dname / "HARDI193.bval", dname / "HARDI193.bvec")
gtab = gradient_table(bvals, bvecs)
```

The data array is shape `(X, Y, Z, N)` where `N` is the number of gradient directions. The `affine` encodes the voxel-to-world-space transform.

### Visualizing the data

Once loaded, it is easy to pull out the b0 (non-diffusion-weighted) volume and plot axial slices with matplotlib:

```python
import matplotlib.pyplot as plt

S0 = data[..., gtab.b0s_mask]
fig, axes = plt.subplots(1, 3)
for i, ax in enumerate(axes):
    ax.imshow(S0[:, :, S0.shape[2] // 2 + i * 5, 0].T, cmap="gray", origin="lower")
plt.savefig("data.png")
```

This tutorial is a starting point before moving on to model fitting and tractography. The output is `data.png` and the raw b0 volume saved as `HARDI193_S0.nii.gz`.
