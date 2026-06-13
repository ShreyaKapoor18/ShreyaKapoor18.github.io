---
title: 'MEG Beamforming: Forward and Inverse Problems'
date: 2026-06-14
permalink: /posts/2026-06-14-meg-beamforming/
tags:
  - Neuroimaging
  - MEG
  - Beamforming
  - Source Localisation
---

## MEG Beamforming: Forward and Inverse Problems

Magnetoencephalography (MEG) measures the tiny magnetic fields (on the order of 100 femtotesla) produced by synchronised postsynaptic currents in cortical neurons. Compared to EEG, MEG is less distorted by the skull and scalp, giving it better spatial resolution, while still offering millisecond temporal precision. The central challenge is: given ~300 sensor measurements on the scalp, where in the brain did the activity come from? This is the inverse problem, and beamforming is one of the most widely used approaches to solve it.

---

## The Forward Problem

The forward problem asks: *given a known current source inside the brain, what signals would we expect to see at the sensors?*

### Neural sources

MEG is sensitive to primary currents — the net intracellular current flow in apical dendrites of pyramidal neurons when they receive synchronised input. These are modelled as **equivalent current dipoles (ECDs)**: a point source with a position **r** and a moment vector **q** (amplitude × orientation).

The magnetic field **B** at a sensor location **r_s** due to a dipole at **r** in a spherical conductor is given by the Sarvas formula:

$$\mathbf{B}(\mathbf{r}_s) = \frac{\mu_0}{4\pi} \frac{\mathbf{q} \times \mathbf{r} \cdot \hat{n}_s}{|\mathbf{r}_s - \mathbf{r}|^3} F(\mathbf{r}, \mathbf{r}_s) - \nabla F \cdot (\mathbf{q} \times \mathbf{r})$$

where F is a geometry-dependent scalar function. In practice, the forward model is computed numerically.

### Head models

Three head models are commonly used, in order of increasing accuracy:

| Model | Description | When to use |
|-------|-------------|-------------|
| Single sphere | Uniform conducting sphere | Quick checks, older datasets |
| Local spheres | One sphere fitted per sensor | MEG, reasonably fast |
| Boundary Element Method (BEM) | Realistic head geometry from MRI | Preferred for source localisation |

A BEM model requires segmenting the MRI into inner skull, outer skull, and scalp surfaces. MNE-Python does this with:

```python
import mne

# create BEM surfaces from the FreeSurfer recon-all output
mne.bem.make_watershed_bem(subject="sub-01", subjects_dir="/path/to/subjects")

# compute BEM solution
bem_model = mne.make_bem_model(subject="sub-01", subjects_dir="/path/to/subjects")
bem_solution = mne.make_bem_solution(bem_model)
```

### The lead field matrix

The forward solution is a **lead field matrix** L of shape `(n_sensors, 3 * n_sources)`. Each column contains the sensor topography that a unit dipole at that source location and orientation would produce. For a source grid with N points and 3 orientations:

```python
src = mne.setup_source_space(subject="sub-01", spacing="oct6",
                             subjects_dir="/path/to/subjects")

fwd = mne.make_forward_solution(
    info=raw.info,
    trans="sub-01-trans.fif",   # MEG-to-MRI coregistration
    src=src,
    bem=bem_solution,
    meg=True,
    eeg=False,
)
```

The coregistration file (`trans`) aligns MEG sensor coordinates to the MRI. This is computed by digitising fiducial landmarks (nasion, left/right pre-auricular points) and fitting them to the MRI surface.

---

## The Inverse Problem

The inverse problem asks: *given the observed sensor data, what brain sources produced it?*

It is fundamentally ill-posed: there are infinitely many source configurations that can produce the same sensor measurements. Different inverse methods impose different constraints to select a unique solution.

### Why it is ill-posed

If **m** is the `(n_sensors, n_times)` measurement matrix and **L** is the lead field, then:

$$\mathbf{m} = \mathbf{L} \mathbf{s} + \epsilon$$

where **s** is the `(n_sources, n_times)` source time series and **ε** is noise. With typically ~300 sensors and ~8000 source dipoles, the system is massively underdetermined. Any **s** = **s₀** + **n** where **Ln** = 0 (i.e. **n** is in the null space of **L**) is also a valid solution.

Common approaches impose different priors:
- **MNE (minimum norm estimate):** L2 penalty — prefers smooth, distributed sources
- **sLORETA:** normalised MNE with zero localisation error for point sources
- **LCMV beamformer:** spatial filter — prefers focal, uncorrelated sources
- **DICS:** frequency-domain beamformer for oscillatory activity

---

## Beamforming

A beamformer is a **spatial filter**: for each candidate source location **r**, it computes a set of weights **w(r)** such that the filtered output

$$\hat{s}(\mathbf{r}, t) = \mathbf{w}(\mathbf{r})^\top \mathbf{m}(t)$$

passes activity from **r** while suppressing interference from all other locations.

### LCMV — Linearly Constrained Minimum Variance

The LCMV beamformer (Van Veen et al., 1997) minimises the output variance subject to a unit-gain constraint on the target dipole:

$$\min_{\mathbf{w}} \; \mathbf{w}^\top \mathbf{C} \mathbf{w} \quad \text{subject to} \quad \mathbf{w}^\top \mathbf{l}(\mathbf{r}) = 1$$

where **C** is the `(n_sensors, n_sensors)` data covariance matrix and **l(r)** is the lead field column for source **r** (assuming a fixed dipole orientation). The closed-form solution is:

$$\mathbf{w}(\mathbf{r}) = \frac{\mathbf{C}^{-1} \mathbf{l}(\mathbf{r})}{\mathbf{l}(\mathbf{r})^\top \mathbf{C}^{-1} \mathbf{l}(\mathbf{r})}$$

The output power at each location gives a **pseudo-T** or **Neural Activity Index (NAI)** map that highlights active sources.

A common contrast is active vs. control covariance (e.g. stimulus period vs. baseline):

$$\text{NAI}(\mathbf{r}) = \frac{\mathbf{w}^\top \mathbf{C}_\text{active} \mathbf{w}}{\mathbf{w}^\top \mathbf{C}_\text{control} \mathbf{w}}$$

### LCMV in MNE-Python

```python
from mne.beamformer import make_lcmv, apply_lcmv

# compute data covariance from active window
data_cov = mne.compute_covariance(epochs, tmin=0.0, tmax=0.3,
                                   method="empirical")

# compute noise covariance from baseline
noise_cov = mne.compute_covariance(epochs, tmin=-0.2, tmax=0.0,
                                    method="empirical")

# make the spatial filter
filters = make_lcmv(
    info=epochs.info,
    forward=fwd,
    data_cov=data_cov,
    noise_cov=noise_cov,
    reg=0.05,           # regularisation: fraction of mean eigenvalue
    pick_ori="max-power",  # choose orientation with max power
)

# apply to evoked response
evoked = epochs.average()
stc = apply_lcmv(evoked, filters)

# plot source time course
stc.plot(subject="sub-01", subjects_dir="/path/to/subjects")
```

### DICS — Dynamic Imaging of Coherent Sources

DICS (Gross et al., 2001) extends beamforming to the frequency domain. Instead of the time-domain covariance matrix, it uses the **cross-spectral density (CSD)** matrix at a frequency of interest:

$$\mathbf{w}_\text{DICS}(\mathbf{r}, f) = \frac{\mathbf{C}_{f}^{-1} \mathbf{l}(\mathbf{r})}{\mathbf{l}(\mathbf{r})^\top \mathbf{C}_{f}^{-1} \mathbf{l}(\mathbf{r})}$$

This makes DICS the method of choice for localising oscillatory sources (e.g. alpha, beta, gamma band activity):

```python
from mne.beamformer import make_dics, apply_dics_csd
from mne.time_frequency import csd_morlet

# compute CSD in the beta band
freqs = [15, 20, 25, 30]
csd = csd_morlet(epochs, frequencies=freqs, tmin=0.0, tmax=0.3, decim=5)

filters = make_dics(
    info=epochs.info,
    forward=fwd,
    csd=csd.mean(),
    reg=0.05,
    pick_ori="max-power",
)

stc, freqs = apply_dics_csd(csd, filters)
stc.plot(subject="sub-01", subjects_dir="/path/to/subjects")
```

---

## Practical considerations

**Regularisation.** The covariance matrix is often ill-conditioned, especially when the number of time samples is not much larger than the number of sensors. A regularisation term `reg * trace(C) / n_sensors` is added to the diagonal. Values between 0.01 and 0.1 are typical.

**Correlated sources.** LCMV assumes sources are uncorrelated. When two regions are synchronised (e.g. left and right motor cortex during bimanual movements), the beamformer suppresses both. Dual-core beamformers or null-beamformer corrections are needed in that case.

**Coregistration accuracy.** Errors in the MEG-to-MRI transform propagate directly into the lead field and degrade localisation. Sub-millimetre coregistration error is the target; always visually inspect the alignment.

**Source space resolution.** A `oct6` source space has ~4098 dipoles per hemisphere at ~4 mm spacing. Finer grids (e.g. `oct7`) increase computation time quadratically via the BEM.

---

## Further reading

- Van Veen et al. (1997). *Localisation of brain electrical activity via linearly constrained minimum variance spatial filtering.* IEEE TBME.
- Gross et al. (2001). *Dynamic imaging of coherent sources.* PNAS.
- [MNE-Python beamformer tutorial](https://mne.tools/stable/auto_tutorials/inverse/50_beamformer_lcmv.html)
- [MNE-Python DICS tutorial](https://mne.tools/stable/auto_tutorials/inverse/70_dics.html)
