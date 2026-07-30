---
title: 'Restricted Boltzmann Machines for Interpretable Neuroimaging'
date: 2026-07-30
permalink: /posts/2026-07-30-rbm-seminar/
tags:
  - Machine Learning
  - Neuroimaging
  - Restricted Boltzmann Machines
  - Deep Belief Networks

---

This post is a write-up of a seminar report I wrote during my M.Sc. in Life Science Informatics at the University of Bonn, for the "Visualization and Medical Image Analysis" seminar at b-it (WiSe 2019/20): *Using Generative-Discriminative Learning in Neuroimaging for Interpretable Predictions*. The full report is linked at the bottom; this is the part of it I still think about most, Restricted Boltzmann Machines, and why a fairly old, physics-flavored idea turned out to be a useful tool for making neuroimaging classifiers less of a black box.

### The problem: accurate but silent classifiers

Machine learning models trained on MRI data can distinguish diseased from healthy subjects with high accuracy, but a discriminative classifier by itself doesn't tell you *why*. It learns a mapping from input to label and is free to lean on whatever numerically convenient pattern separates the classes, whether or not that pattern means anything neurobiologically. For a research question ("which brain connections break down in aphasia?") or a clinical one ("what changes as Huntington's disease progresses?"), that's not good enough.

The fix explored in the report is to put a *generative* model in front of the *discriminative* one. Instead of feeding raw voxels straight into a classifier, first learn a lower-dimensional, more meaningful representation of the data, then classify in that space. Two very different ways of doing this show up in the neuroimaging literature: one that encodes prior neurobiological knowledge (Dynamic Causal Models, used with fMRI), and one that learns its representation with no prior knowledge at all: Restricted Boltzmann Machines, used with structural MRI.

### What an RBM actually is

An RBM is a generative neural network with exactly two layers, visible and hidden, connected symmetrically, with no connections within either layer. Voxel data goes into the visible units; the hidden units act as learned, non-linear feature detectors. The whole thing is defined through an energy function over a visible vector *v* and hidden vector *h*:

$$E(v,h) = -\sum_{i,j} w_{ij} h_i v_j - \sum_i c_i h_i - \sum_j b_j v_j$$

with weights *w*, hidden biases *c*, and visible biases *b*. The joint distribution is $p(v,h) = \frac{1}{Z} e^{-E(v,h)}$, where *Z* is the partition function, directly borrowed from the Boltzmann distribution in statistical mechanics. Minimizing energy is the same as maximizing probability, so training the network means finding the weights that make the observed data configurations low-energy.

Because there are no within-layer connections, the conditional distributions factorize cleanly:

$$p(h_i=1\mid v) = \frac{1}{1+e^{-c_i - \sum_j w_{ij}v_j}}, \qquad p(v_j=1\mid h) = \frac{1}{1+e^{-b_j - \sum_i w_{ij}h_i}}$$

which is what makes block Gibbs sampling tractable: sample *h* given *v*, then *v* given *h*, back and forth. Training uses Contrastive Divergence, an approximation to the true log-likelihood gradient that only requires running this Gibbs chain a small number of steps *k* rather than to convergence. A Deep Belief Network is simply a stack of RBMs, trained greedily layer by layer (each layer's hidden output becomes the next layer's visible input), then fine-tuned end-to-end with backpropagation once a classification layer (softmax, in this case) is attached on top.

### Two case studies from the report

**Generative embedding with prior knowledge (fMRI, aphasia).** Brodersen et al. (2011) fit a Dynamic Causal Model of the thalamo-temporal cortex to speech-processing fMRI data from 37 subjects (26 healthy, 11 with aphasia), then used the DCM's fitted connection-strength parameters, not raw voxel activity, as the feature space for an SVM. This "generative embedding" beat both a searchlight approach (73.0% accuracy) and PCA-based dimensionality reduction (86.5%), reaching 97.3% balanced accuracy, and it pointed to specific right-to-left hemisphere connections as the discriminative ones, consistent with known lateralization of language processing.

**RBMs/DBNs with no prior knowledge (sMRI, Huntington's and schizophrenia).** Plis et al. (2014) went the other direction: no neurobiological model, just a DBN learning directly from structural MRI voxels. On a Huntington's disease dataset (3,500 scans), a 3-layer DBN's hidden representation, visualized in 2D with a divide-and-conquer embedding, separated healthy from diseased subjects and recovered a gradient corresponding to disease severity, despite the network only ever being trained on binary healthy/diseased labels. On a combined schizophrenia dataset (198 patients, 191 controls), classification F-scores rose consistently with DBN depth: an SVM went from 0.68 on raw data to 0.90 with a 3-layer DBN as the feature extractor; logistic regression from 0.63 to 0.91; kNN from 0.61 to 0.90.

| | Generative embedding (Brodersen et al., 2011) | RBM / DBN (Plis et al., 2014) |
|---|---|---|
| Modality | fMRI | structural MRI |
| Generative model | Dynamic Causal Model, hand-specified | RBM/DBN, learned, no prior structure |
| Fit per | subject | shared across whole dataset |
| Embedding / features | fitted connectivity parameters | hidden-unit activations |
| Prior neurobiological knowledge required | yes | no |
| Interpretability | high, maps onto known circuits | low, needs further probing |
| Classifier tested | SVM | SVM, logistic regression, kNN |
| Reported result | 97.3% balanced accuracy | F = 0.90 (SVM, 3-layer DBN) |

### The trade-off

The report's conclusion still holds up: generative embeddings are "semi-automatic" feature learning, useful precisely when you already have a plausible model of the underlying biology to encode (which region connects to which, for instance), while DBNs/RBMs are "fully automatic," useful when no such prior exists and you want the network to discover structure on its own, at the cost of needing more data and giving up some of that built-in interpretability.

Looking back, though, the two case studies aren't really a controlled comparison, since they differ in modality, disease, and dataset, so I can't say one approach beat the other. If I were to redo the seminar report, I would pick the same modality and dataset for both approaches and then perform a proper comparison.

No single study runs that comparison directly, but two more recent, same-modality papers get close to what I'd want to put side by side. On the fully-automatic side, Yan et al. (2021) train a deep network directly on raw fMRI time series, no generative model in between, to classify schizophrenia, bipolar disorder, and schizoaffective disorder. On the generative-embedding side, Galioulline et al. (2023) fit a regression DCM to resting-state fMRI and use the fitted connectivity parameters, not raw voxels, to predict future depressive episodes in a large UK Biobank sample. Both fMRI, both recent, one with a hand-specified generative model in the pipeline and one without, which is roughly the controlled comparison the original report was missing. Hjelm et al. (2014), already in the references below but not one of the case studies above, is an earlier example of the same fully-automatic, fMRI-based idea: RBMs applied directly to fMRI to identify intrinsic functional networks.

What keeps pulling me back to RBMs specifically is that they're one of the last widely-used architectures trained by something other than end-to-end backpropagation through a single loss, Gibbs sampling and contrastive divergence instead, straight out of statistical mechanics, doing representation learning before "representation learning" was the standard vocabulary for it. That's also roughly the intuition behind Hopfield networks and other energy-based models: define an energy landscape, let training reshape it, let inference be a walk downhill.

### The original seminar report

<p style="text-align: center;">
  <a href="/files/ShreyaKapoor_Seminar.pdf" target="_blank">Download the full seminar report (PDF)</a>
</p>

### References

- Fischer, A., & Igel, C. An introduction to restricted Boltzmann machines.
- Hinton, G. E. Deep belief nets.
- Hjelm, R. D., Calhoun, V. D., Salakhutdinov, R., Allen, E. A., Adali, T., & Plis, S. M. (2014). Restricted Boltzmann machines for neuroimaging: an application in identifying intrinsic networks. *NeuroImage*, 96, 245–260.
- Plis, S. M., Hjelm, D. R., Salakhutdinov, R., Allen, E. A., Bockholt, H. J., Long, J. D., Johnson, H. J., Paulsen, J. S., Turner, J. A., & Calhoun, V. D. (2014). Deep learning for neuroimaging: a validation study. *Frontiers in Neuroscience*, 8, 229.
- Brodersen, K. H., Schofield, T. M., Leff, A. P., Ong, C. S., Lomakina, E. I., Buhmann, J. M., & Stephan, K. E. (2011). Generative embedding for model-based classification of fMRI data. *PLOS Computational Biology*, 7(6), e1002079.
- Yan, W., Zhao, M., Fu, Z., Pearlson, G. D., Sui, J., & Calhoun, V. D. (2021). Mapping relationships among schizophrenia, bipolar and schizoaffective disorders: a deep classification and clustering framework using fMRI time series. *Schizophrenia Research*, 245, 141–150.
- Galioulline, H., Frässle, S., Harrison, S. J., Pereira, I., Heinzle, J., & Stephan, K. E. (2023). Predicting future depressive episodes from resting-state fMRI with generative embedding. *NeuroImage*, 273, 119986.
