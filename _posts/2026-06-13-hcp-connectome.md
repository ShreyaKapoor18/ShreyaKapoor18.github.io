---
title: 'Generating Structural Connectomes from the Human Connectome Project'
date: 2026-06-13
permalink: /posts/2026-06-13-hcp-connectome/
tags:
  - Neuroimaging
  - Diffusion MRI
  - Connectome
  - MRtrix
  - HCP
---

## Generating Structural Connectomes from the Human Connectome Project

The [Human Connectome Project (HCP)](https://www.humanconnectomeproject.org/) provides high-quality diffusion MRI data for over 1,000 healthy young adults, acquired at 3T with 1.25 mm isotropic resolution and 90 gradient directions at b=3000 s/mm². These data are well-suited for building structural connectomes: graphs that quantify white matter connectivity between brain regions. This post walks through the full pipeline using MRtrix3 and FSL.

### What is a structural connectome?

A structural connectome is a graph where:
- **Nodes** are brain regions defined by a parcellation atlas (e.g. the 84-region FreeSurfer Desikan-Killiany atlas)
- **Edges** represent white matter fiber bundles connecting pairs of regions, reconstructed via tractography
- **Edge weights** encode connection strength — typically the number of SIFT-filtered streamlines, mean FA, or streamline density normalised by region volume

The result is a symmetric matrix per subject, which can then be used for downstream analyses such as network statistics, group comparisons, or machine learning.

### Required HCP files

For each subject the pipeline requires five files from the HCP S1200 release (available via ConnectomeDB after data use agreement):

| File | Description |
|------|-------------|
| `data.nii.gz` | Raw 4D diffusion-weighted image |
| `bvals` / `bvecs` | Gradient encoding table |
| `nodif_brain_mask.nii.gz` | Binary brain mask |
| `aparc+aseg.nii.gz` | FreeSurfer parcellation + segmentation |
| `T1w_acpc_dc_restore_brain.nii.gz` | Skull-stripped T1 in diffusion space |

### Step 1: Response function estimation

CSD requires an estimate of the single-fiber response function. The `dhollander` algorithm automatically identifies WM, GM, and CSF voxels without a manual ROI:

```bash
dwi2response dhollander data.nii.gz \
  response_wm.txt response_gm.txt response_csf.txt \
  -fslgrad bvecs bvals \
  -mask nodif_brain_mask.nii.gz
```

Three response functions are estimated — one per tissue class. This enables multi-tissue CSD in the next step, which handles partial volume much better than single-tissue CSD, especially near the cortex.

### Step 2: Multi-tissue CSD (MSMT-CSD)

`dwi2fod msmt_csd` jointly models WM, GM, and CSF signal fractions to produce sharper fiber orientation distributions (FODs):

```bash
dwi2fod msmt_csd data.nii.gz \
  response_wm.txt wm_fod.mif \
  response_gm.txt gm_fod.mif \
  response_csf.txt csf_fod.mif \
  -fslgrad bvecs bvals \
  -mask nodif_brain_mask.nii.gz
```

The WM FOD image is the primary input for tractography. You can inspect the FODs visually with `mrview wm_fod.mif -odf.load_sh wm_fod.mif`.

### Step 3: Five-tissue-type segmentation (5TT)

Anatomically Constrained Tractography (ACT) requires a five-tissue-type image that tells the tracker where it is anatomically allowed to be. MRtrix generates this from the T1 using FSL:

```bash
5ttgen fsl T1w_acpc_dc_restore_brain.nii.gz 5tt.mif
```

The five classes are: cortical GM, subcortical GM, WM, CSF, and pathological tissue. Streamlines are terminated when they enter CSF or leave the brain mask, and are only accepted if they end in GM.

### Step 4: Probabilistic tractography with ACT

We generate 1 million streamlines using dynamic seeding, which places seeds proportional to the WM FOD amplitude rather than uniformly in a mask:

```bash
tckgen wm_fod.mif tractogram_1M.tck \
  -act 5tt.mif \
  -backtrack \
  -seed_dynamic wm_fod.mif \
  -maxlength 250 \
  -select 1000000
```

`-backtrack` allows the tracker to reverse and try a different direction if it enters an anatomically invalid region, which substantially reduces premature terminations.

### Step 5: SIFT filtering

SIFT (Spherical-deconvolution Informed Filtering of Tractograms) removes streamlines so that the remaining tractogram is consistent with the underlying FOD amplitudes across the brain. Without SIFT, longer fibers tend to be systematically underrepresented and short U-fibers overrepresented:

```bash
tcksift tractogram_1M.tck wm_fod.mif tractogram_SIFT.tck \
  -act 5tt.mif
```

After SIFT the streamline count per bundle is a more reliable proxy for connection strength and is more comparable across subjects.

### Step 6: Parcellation and connectome construction

The FreeSurfer `aparc+aseg` label image uses integer codes that need to be remapped to a compact node index. MRtrix ships with a lookup table for the Desikan-Killiany atlas:

```bash
labelconvert aparc+aseg.nii.gz \
  $FREESURFER_HOME/FreeSurferColorLUT.txt \
  $(dirname $(which mrconvert))/../share/mrtrix3/labelconvert/fs_default.txt \
  parcellation.mif
```

Then assign each streamline endpoint to a parcel and count connections:

```bash
tck2connectome tractogram_SIFT.tck parcellation.mif connectome.csv \
  -scale_invnodevol \
  -symmetric \
  -zero_diagonal
```

`-scale_invnodevol` divides each edge weight by the inverse of both endpoint node volumes, which corrects for the tendency of larger regions to attract more streamlines simply because of their size.

The output is an 84×84 symmetric CSV matrix — the structural connectome for one subject.

### Reading the connectome in Python

```python
import numpy as np

conn = np.loadtxt("connectome.csv", delimiter=",")  # shape (84, 84)

# extract upper triangle as a feature vector (3570 values)
mat = np.triu_indices(84, k=1)
features = conn[mat]
```

Stacking feature vectors across subjects gives a matrix of shape `(n_subjects, 3570)` ready for downstream analyses such as graph-theoretic measures, dimensionality reduction, or classification.

### Further reading

- [MRtrix ISMRM HCP tutorial](https://mrtrix.readthedocs.io/en/latest/quantitative_structural_connectivity/ismrm_hcp_tutorial.html)
- [HCP data access and citation requirements](https://www.humanconnectome.org/study/hcp-young-adult/document/hcp-citations)
- [FSL FDT user guide](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT/UserGuide)
