---
title: 'Generating Structural Connectomes from the Human Connectome Project'
date: 2026-06-14
permalink: /posts/2026-06-14-hcp-connectome/
tags:
  - Neuroimaging
  - Diffusion MRI
  - Connectome
  - MRtrix
  - HCP
---

## Generating Structural Connectomes from the Human Connectome Project

The [Human Connectome Project (HCP)](https://www.humanconnectomeproject.org/) provides high-quality diffusion MRI data for over 1,000 healthy young adults. In my master's thesis I used these data to build structural connectomes for each subject and then extract predictive subgraphs related to personality traits. This post walks through the connectome generation pipeline.

### What is a structural connectome?

A structural connectome is a graph where:
- **Nodes** are brain regions defined by a parcellation atlas (we use the 84-region FreeSurfer Desikan-Killiany atlas)
- **Edges** represent white matter fiber bundles connecting pairs of regions
- **Edge weights** encode the strength of the connection, for example the number of streamlines or the mean FA along the bundle

The result is an 84×84 symmetric matrix per subject.

### Required HCP files

For each subject the pipeline needs five files from the HCP S1200 release:

| File | Description |
|------|-------------|
| `data.nii.gz` | Raw 4D diffusion-weighted image |
| `bvals` / `bvecs` | Gradient encoding table |
| `nodif_brain_mask.nii.gz` | Binary brain mask |
| `aparc+aseg.nii.gz` | FreeSurfer parcellation + segmentation |
| `T1w_acpc_dc_restore_brain.nii.gz` | Skull-stripped T1 in diffusion space |

### Pipeline overview

The preprocessing follows the [MRtrix ISMRM HCP tutorial](https://mrtrix.readthedocs.io/en/latest/quantitative_structural_connectivity/ismrm_hcp_tutorial.html) with FSL 5.0 for auxiliary steps.

![Connectome generation pipeline](../../images/Connectome_generation_pipeline.png)

#### Step 1: Response function estimation

Before running CSD we estimate the single-fiber response function from the data using `dwi2response`:

```bash
dwi2response dhollander data.nii.gz response_wm.txt response_gm.txt response_csf.txt \
  -fslgrad bvecs bvals -mask nodif_brain_mask.nii.gz
```

The `dhollander` algorithm automatically selects WM, GM, and CSF voxels to estimate three-tissue response functions — no manual ROI needed.

#### Step 2: Multi-tissue CSD

Constrained spherical deconvolution (CSD) estimates fiber orientation distributions (FODs) at each voxel. Using multi-tissue CSD (`dwi2fod msmt_csd`) jointly models WM, GM, and CSF signals, which improves tracking near the cortex where partial volume is strong:

```bash
dwi2fod msmt_csd data.nii.gz \
  response_wm.txt wm_fod.mif \
  response_gm.txt gm_fod.mif \
  response_csf.txt csf_fod.mif \
  -fslgrad bvecs bvals -mask nodif_brain_mask.nii.gz
```

#### Step 3: Five-tissue-type segmentation (5TT)

MRtrix uses a five-tissue-type (5TT) image to constrain tractography anatomically. We generate it from the T1 using `5ttgen`:

```bash
5ttgen fsl T1w_acpc_dc_restore_brain.nii.gz 5tt.mif
```

This produces a 4D image with one volume per tissue class: cortical GM, subcortical GM, WM, CSF, and pathological tissue. Streamlines that enter CSF or exit the brain are terminated.

#### Step 4: Probabilistic tractography with ACT

We seed 1 million streamlines using Anatomically Constrained Tractography (ACT):

```bash
tckgen wm_fod.mif tractogram_1M.tck \
  -act 5tt.mif \
  -backtrack \
  -seed_dynamic wm_fod.mif \
  -maxlength 250 \
  -select 1000000
```

Dynamic seeding (`-seed_dynamic`) places seeds proportional to the WM FOD amplitude, which reduces the bias toward short fibers that fixed-density seeding introduces.

![1M streamline tractogram](../../images/tractography_1M.png)

#### Step 5: SIFT filtering

1 million streamlines is a lot, but many will be false positives. SIFT (Spherical-deconvolution Informed Filtering of Tractograms) removes streamlines so that the remaining tractogram is consistent with the underlying FOD amplitudes:

```bash
tcksift tractogram_1M.tck wm_fod.mif tractogram_SIFT.tck -act 5tt.mif
```

After SIFT the tractogram is more biologically plausible and edge weights computed from it are more comparable across subjects.

#### Step 6: Parcellation and connectome construction

The FreeSurfer `aparc+aseg` label image is mapped to MRtrix format and used to assign each streamline endpoint to a brain region:

```bash
labelconvert aparc+aseg.nii.gz \
  $FREESURFER_HOME/FreeSurferColorLUT.txt \
  $MRTRIX/share/mrtrix3/labelconvert/fs_default.txt \
  parcellation.mif

tck2connectome tractogram_SIFT.tck parcellation.mif connectome.csv \
  -scale_invnodevol \
  -symmetric \
  -zero_diagonal
```

`-scale_invnodevol` normalises edge weights by the inverse of the node volumes, making the matrix comparable across subjects with different brain sizes.

The output is an 84×84 CSV matrix — one structural connectome per subject.

![Structural connectome matrix](../../images/connectome_1M.png)

### Using the connectomes

Once all subjects have been processed, we read each matrix into a `BrainGraph` object (a NetworkX subclass) and stack them into a feature matrix:

```python
import numpy as np
import pandas as pd

num_nodes = 84
mat = np.triu_indices(num_nodes)  # upper triangular indices

# for each subject, read the CSV and extract upper triangle
rows = []
for subject_id in subject_list:
    conn = np.loadtxt(f"{subject_id}/connectome.csv", delimiter=",")
    rows.append(conn[mat])

X = pd.DataFrame(rows)  # shape: (n_subjects, n_edges)
```

Each row is a flattened upper triangle of the 84×84 matrix, giving 3570 edge features per subject. These features were then used for downstream classification of personality traits (Big Five) and gender using XGBoost, SVC, Random Forest, and MLP classifiers — but that is a story for another post.
