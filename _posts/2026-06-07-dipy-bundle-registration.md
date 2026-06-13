---
title: 'Groupwise Bundle Registration with DIPY'
date: 2026-06-07
permalink: /posts/2026-06-07-dipy-bundle-registration/
tags:
  - Neuroimaging
  - DIPY
  - Tractography
  - Diffusion MRI
---

## Groupwise Bundle Registration

When comparing white matter bundles across subjects, they first need to be coregistered to a common space. DIPY's `groupwise_slr` function does this without requiring a pre-defined atlas: it iteratively aligns all bundles to an unbiased mean shape using Streamline Linear Registration (SLR).

### What is SLR?

SLR (Streamline Linear Registration) optimizes a linear transform (translation, rotation, scaling, shearing) that minimizes the average distance between pairs of streamlines across two tractograms. The distance metric used is the minimum average direct-flip (MDF) distance, which is robust to streamline direction.

### Loading bundles

The tutorial loads five left arcuate fasciculi from different subjects (or bundle samples):

```python
from dipy.data import get_two_hemi_templates
from dipy.segment.bundles import groupwise_slr

# load 5 bundle instances as lists of streamline arrays
bundles = [...]  # list of StatefulTractogram objects
```

### Running groupwise registration

```python
from dipy.segment.bundles import groupwise_slr

bundles_reg, aff_list, d_list = groupwise_slr(
    bundles,
    nb_pts=20,
    verbose=True
)
```

The function returns:
- `bundles_reg` — the aligned bundle list
- `aff_list` — the per-subject linear transforms
- `d_list` — the pairwise distance at each iteration

The pairwise distance typically drops noticeably in the first few iterations and then plateaus, indicating convergence.

### Before and after

![Before registration](../../images/before_group_registration.png)
![After registration](../../images/after_group_registration.png)

The left panel shows five bundles before alignment; they occupy different parts of space. After `groupwise_slr`, all five overlap in a common coordinate frame, making cross-subject comparisons meaningful.
