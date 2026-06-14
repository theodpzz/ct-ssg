<p align="center">
  <h2 align="center">Structured Spectral Graph Representation Learning for Multi-label Abnormality Analysis from 3D CT Scan  🩺👨🏻‍⚕️</h2>
  <h4 align="center"><b>MELBA 2026</b></h4>
  <p align="center">
    <a href="http://arxiv.org/pdf/2510.10779"><img alt='arXiv' src="https://img.shields.io/badge/arXiv-2510.10779-b31b1b.svg"></a>
    <a href="https://youtu.be/00dRw8bX4lM"><img alt="YouTube" src="https://img.shields.io/badge/YouTube-Video-red?logo=youtube"></a>
    <a href="https://huggingface.co/theodpzz/ct-ssg"><img alt="Weights" src="https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface"></a>
  </a>
  </p>
</p>

---

## 🧩 Method Overview

(1) Adjacent axial slices are grouped into triplets, each representing a node in a graph. (2) Edges between nodes are weighted according to their physical distance along the z-axis. (3) Node features are enhanced with Triplet Axial Slices positional embeddings, and then processed by a Spectral Block that incorporates Chebyshev graph convolution for structured spectral modeling. (4) The resulting node representations are aggregated via mean pooling and passed to a classification head to predict abnormalities.

<img src="./figures/method_overview.png" alt="Method overview" width="900">

  > #### **Structured Spectral Graph Representation Learning for Multi-label Abnormality Analysis from 3D CT Scan**<be>  
  >Machine Learning for Biomedical Imaging Journal (MELBA), 2026.
  >Theo Di Piazza, Carole Lazarus, Olivier Nempont, Loic Boussel.
---

### Notice

This repository is currently under review for compliance with institutional and collaborative agreements.

The repository will be made publicly available once the approval process is completed.

---

## 🤝🏻 Acknowledgment

We thank contributors from the CT-RATE dataset available at [https://huggingface.co/datasets/ibrahimhamamci/CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE), from the Rad-ChestCT dataset available at [https://zenodo.org/records/6406114](https://zenodo.org/records/6406114) and from the Merlin Abdominal CT dataset available at [https://stanfordaimi.azurewebsites.net/categories/datasets?domain=BODY](https://stanfordaimi.azurewebsites.net/categories/datasets?domain=BODY).

---

## Purpose

This code is provided for **academic and research purposes only**, to support reproducibility of the results described in the associated paper. This repository is a research prototype, and is not intended for clinical use.

---

## 📎Citation

If you find this repository useful for your work, we would appreciate the following citation:

```bibtex
@article{dipiazza_ssg_2026,
    title   = "Structured Spectral Graph Representation Learning for Multi-label Abnormality Analysis from 3D CT Scans",
    author  = "Di Piazza, Theo and Lazarus, Carole and Nempont, Olivier and Boussel, Loic",
    journal = "Machine Learning for Biomedical Imaging",
    volume  = "2026",
    issue   = "June 2026 issue",
    year    = "2026",
    pages   = "359--388",
    issn    = "2766-905X",
    doi     = "https://doi.org/10.59275/j.melba.2026-87e3",
    url     = "https://melba-journal.org/"
}

```
