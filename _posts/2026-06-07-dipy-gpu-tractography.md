---
title: 'GPU-Accelerated Tractography with DIPY and cuslines'
date: 2026-06-07
permalink: /posts/2026-06-07-dipy-gpu-tractography/
tags:
  - Neuroimaging
  - DIPY
  - Tractography
  - GPU
  - Diffusion MRI
---

## GPU-Accelerated Tractography

For whole-brain tractography the bottleneck is usually the inner tracking loop: at each step you evaluate a model, sample a direction, and move the streamline. DIPY supports offloading this to GPU via the [cuslines](https://github.com/dipy/cuslines) library, which provides CUDA, Metal, and WebGPU backends alongside a CPU fallback using numba.

### Supported backends

| Backend | Hardware |
|---------|----------|
| `gpu` (CUDA) | NVIDIA GPU |
| `metal` | Apple Silicon / AMD GPU on macOS |
| `webgpu` | Cross-platform via wgpu |
| `cpu` | CPU with numba JIT |

The backend is selected at runtime via `--device`, so the same script runs on any hardware.

### Usage

The script accepts a DWI image, b-values, b-vectors, a brain mask, and an ROI seed mask:

```bash
python run_gpu_streamlines.py dwi.nii.gz bvals bvecs mask.nii.gz roi.nii.gz \
  --device gpu \
  --dg prob \
  --output-prefix results/tractogram
```

If no files are provided it defaults to the Stanford HARDI dataset for easy testing.

### How it works

Internally, `cuslines` takes the CSD or DTI peaks array and seeds, and runs the tracking loop in parallel across seeds on device. The Python side only needs to:

1. Fit the ODF model on CPU (or GPU if supported)
2. Pass the peaks array and seed coordinates to `cuslines`
3. Receive the streamline array back in host memory

```python
import cuslines

gpu_tracker = cuslines.GPUTracker(
    data=peaks_array,
    seeds=seed_coords,
    affine=affine,
    step_size=0.5,
    max_length=300,
    device="gpu"
)
streamlines = gpu_tracker.generate()
```

On a modern GPU this can be 10-100x faster than single-threaded CPU tracking, making whole-brain probabilistic tractography with millions of seeds practical in minutes rather than hours.
