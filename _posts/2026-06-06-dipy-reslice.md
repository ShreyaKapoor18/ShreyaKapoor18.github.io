---
title: 'Reslicing Diffusion MRI Datasets to Isotropic Voxels'
date: 2026-06-06
permalink: /posts/2026-06-06-dipy-reslice/
tags:
  - Neuroimaging
  - DIPY
  - Diffusion MRI
---

## Reslicing Datasets

Many diffusion MRI acquisitions use anisotropic voxels, for example `2 x 2 x 3 mm`. This can cause issues downstream in tractography and registration algorithms that assume isotropic spatial resolution. DIPY provides a one-function solution via `dipy.align.reslice`.

### Why isotropic voxels matter

Fiber tracking algorithms sample ODFs along streamline steps. If the voxel grid is anisotropic the step size effectively differs along each axis, which can introduce directional bias into the tractogram. Reslicing to isotropic voxels before tracking removes this asymmetry.

### Reslicing with DIPY

```python
from dipy.align.reslice import reslice
from dipy.io.image import load_nifti, save_nifti

data, affine, voxel_size = load_nifti("dwi.nii.gz", return_voxsize=True)

new_voxel_size = (2.0, 2.0, 2.0)
data_resampled, new_affine = reslice(data, affine, voxel_size, new_voxel_size)

save_nifti("iso_vox.nii.gz", data_resampled, new_affine)
```

`reslice` uses trilinear interpolation by default, which is a good trade-off between speed and accuracy for structural data. The output affine is updated to reflect the new voxel dimensions.

### Saving in multiple formats

DIPY also supports writing the output in SPM Analyze (`.img/.hdr`) format:

```python
import nibabel as nib

img = nib.Spm2AnalyzeImage(data_resampled.astype("float32"), new_affine)
nib.save(img, "iso_vox.img")
```

This is useful when the data will be fed into FSL or SPM pipelines that expect a specific format.
