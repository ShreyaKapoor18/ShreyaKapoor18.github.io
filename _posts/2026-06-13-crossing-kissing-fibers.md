---
title: 'Crossing and Kissing Fibers in Fiber Tractography'
date: 2026-06-13
permalink: /posts/2026-06-13-crossing-kissing-fibers/
tags:
  - Neuroimaging
  - DIPY
  - Tractography
  - Diffusion MRI
---

## Crossing and Kissing Fibers in Fiber Tractography

One of the central challenges in diffusion MRI tractography is that a single imaging voxel is 2–3 mm on a side, yet contains thousands of axons that may run in completely different directions. Two of the most commonly discussed fiber configurations are **crossing fibers** and **kissing fibers**. Understanding the difference between them — and why tractography algorithms struggle to tell them apart — is essential for interpreting any tractogram.

### Crossing vs. Kissing Fibers

**Crossing fibers** occur when two distinct fiber bundles pass through the same voxel in different directions, forming an X-shaped geometry. The bundles genuinely intersect and continue past each other.

**Kissing fibers** occur when two bundles approach each other, touch or come very close within the same voxel, and then curve away in the same direction, forming a U-shaped or tangential geometry. The bundles do not cross; they merely run alongside each other momentarily.

From a diffusion MRI signal alone the two configurations can look identical: both produce a voxel with apparent signal from multiple orientations. A tractography algorithm that cannot distinguish them risks either (a) following the wrong branch at a crossing, or (b) incorrectly treating a genuine kissing configuration as a crossing and generating spurious streamlines that jump between bundles.

### Why DTI Fails Here

Diffusion Tensor Imaging (DTI) fits a single 3D ellipsoid per voxel. This model has exactly one principal eigenvector, which represents the dominant fiber direction. When two populations of fibers cross inside a voxel, the tensor is pulled in two directions simultaneously and the resulting shape becomes oblate or spherical — it no longer points cleanly along any fiber.

```python
import numpy as np
from dipy.reconst.dti import TensorModel
from dipy.data import get_fnames
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames("stanford_hardi")
data, affine = load_nifti(hardi_fname)
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs)

tensor_model = TensorModel(gtab)
tensor_fit = tensor_model.fit(data)

fa = tensor_fit.fa
evecs = tensor_fit.evecs  # principal eigenvectors
```

In regions of fiber crossing, `fa` drops markedly even when both bundles are coherent. The principal eigenvector `evecs[..., 0]` points somewhere between the two true fiber directions — a biologically meaningless average. EuDX and other deterministic algorithms that follow this single direction will simply stall or deviate at every crossing.

![DTI FA map with low-FA crossing regions](../../images/dti_fa_crossings.png)

### Synthetic Phantom: Seeing the Problem Directly

DIPY's `phantoms` module lets us construct a ground-truth crossing geometry and verify that DTI fails while higher-order models succeed.

```python
from dipy.sims.phantom import orbital_phantom
from dipy.sims.voxel import multi_tensor, all_tensor_evecs
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere

sphere = get_sphere("repulsion724")

# Two crossing fiber populations at 90 degrees
mevals = np.array([[0.0015, 0.0003, 0.0003],
                   [0.0015, 0.0003, 0.0003]])

# First bundle along x, second along y
angles = [(90, 0), (90, 90)]
fractions = [50, 50]

signal, sticks = multi_tensor(gtab, mevals, S0=100,
                              angles=angles,
                              fractions=fractions,
                              snr=None)
```

`sticks` contains the ground-truth fiber directions. We can compare them against the peaks recovered by DTI and CSD below.

![Synthetic crossing phantom geometry](../../images/phantom_crossing_geometry.png)

### CSD Resolves Multiple Fiber Populations

Constrained Spherical Deconvolution (CSD) treats the diffusion signal as a convolution of a single-fiber response function with the fiber ODF (fODF). Deconvolving it yields a sharp ODF with distinct peaks, one per fiber population. Unlike DTI, CSD can return multiple peaks per voxel, making fiber crossings tractable.

```python
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.direction import peaks_from_model
from dipy.data import default_sphere

response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=8)
csd_fit = csd_model.fit(data)

csd_peaks = peaks_from_model(
    csd_model, data, default_sphere,
    relative_peak_threshold=0.5,
    min_separation_angle=25,
    return_odf=True,
    npeaks=3,
)
```

Setting `npeaks=3` allows the algorithm to report up to three distinct fiber orientations per voxel. In a region with two crossing bundles the two dominant peaks will align with the true fiber directions; in a kissing configuration only one dominant direction will be present even though the FA is low, which is a useful diagnostic.

![CSD fODF peaks in crossing voxel vs kissing voxel](../../images/csd_peaks_crossing_vs_kissing.png)

### Kissing vs. Crossing: Diagnosing from the fODF

Once CSD peaks are available, a simple heuristic distinguishes the two configurations:

| Configuration | Number of peaks above threshold | Peak angle separation |
|---|---|---|
| Single fiber | 1 | — |
| Crossing fibers | ≥ 2 | typically 30°–90° |
| Kissing fibers | 1 (or 2 with similar directions) | < 30° |

```python
# Identify voxels with two well-separated peaks
n_peaks = (csd_peaks.peak_values > 0.1).sum(axis=-1)
angle_sep = np.degrees(
    np.arccos(np.clip(
        np.abs(np.einsum('...i,...i', csd_peaks.peak_dirs[..., 0, :],
                                     csd_peaks.peak_dirs[..., 1, :])), -1, 1))
)

crossing_mask = (n_peaks >= 2) & (angle_sep > 30)
kissing_mask  = (n_peaks == 1) | ((n_peaks >= 2) & (angle_sep <= 30))
```

This classification is heuristic and sensitive to the `relative_peak_threshold` used in `peaks_from_model`. Regions near the boundary of the two masks should be interpreted cautiously.

![Crossing vs. kissing classification map](../../images/crossing_kissing_mask.png)

### Probabilistic Tracking Through Crossings

Deterministic algorithms commit to a single direction at each step, so they cannot explore both branches of a crossing. Probabilistic algorithms sample from the full fODF at each step, naturally distributing streamlines across all plausible directions.

```python
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.direction import ProbabilisticDirectionGetter
from dipy.tracking import utils

pmf = csd_fit.odf(default_sphere).clip(min=0)

prob_dg = ProbabilisticDirectionGetter.from_pmf(
    pmf, max_angle=30.0, sphere=default_sphere
)

white_matter = fa > 0.2
stopping_criterion = ThresholdStoppingCriterion(fa, 0.2)
seeds = utils.seeds_from_mask(white_matter, affine, density=2)

streamlines = Streamlines(
    LocalTracking(prob_dg, stopping_criterion, seeds,
                  affine=affine, step_size=0.5)
)
```

By sampling directions proportionally to the fODF amplitude, the tracker sends streamlines down both arms of a true crossing while producing a coherent bundle through a kissing region where the fODF has only one dominant lobe.

![Probabilistic tractogram through crossing region](../../images/prob_tractogram_crossings.png)

### Saving and Inspecting the Result

```python
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.streamline import save_tractogram
from dipy.io.image import load_nifti_data
import nibabel as nib

ref_img = nib.load(hardi_fname)
sft = StatefulTractogram(streamlines, ref_img, Space.RASMM)
save_tractogram(sft, "tractogram_crossing_aware.trx")
```

Load the `.trx` file in TrackVis, MI-Brain, or DIPY's own `horizon` viewer to color streamlines by local direction. Crossing bundles will appear as two distinct color populations within the same voxel cluster; kissing bundles will share a color, curving away together.

### Summary

Crossing and kissing fibers are geometrically distinct but produce similar low-FA signatures in DTI, making them easy to confuse. CSD resolves this by recovering multiple fODF peaks per voxel: two well-separated peaks indicate a crossing, a single dominant peak (even in a low-FA region) suggests kissing geometry. Probabilistic direction getters leverage the full fODF distribution to propagate streamlines through crossings without committing to one branch, substantially reducing the number of prematurely terminated or anatomically incorrect tracks.
