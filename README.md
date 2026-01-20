<p align="center">
  <h2 align="center">Structured Spectral Graph Representation Learning for Multi-label Abnormality Analysis from 3D CT Scan 🩺👨🏻‍⚕️</h2>
</p>

PyTorch [CT-SSG](http://arxiv.org/abs/2510.10779) model implementation.

## Method Overview

Adjacent axial slices are grouped into triplets, each representing a node in a graph. Edges between nodes are weighted according to their physical distance along the z-axis. Node features are enhanced with Triplet Axial Slices positional embeddings, and then processed by a Spectral Block that incorporates Chebyshev graph convolution for structured spectral modeling. The resulting node representations are aggregated via mean pooling and passed to a classification head to predict anomalies.

<img src="https://github.com/theodpzz/ssg/blob/master/figures/method_overview.png" alt="Method overview" width="900">

## Getting Started

### Clone the Repository

To clone this repository, use the following command:

```bash
git clone https://github.com/theodpzz/ssg.git
```

### Installation

Make sure you have Python 3 installed. Then, install the dependencies using:

```bash
pip install -r requirements.txt
```

### Quick start

```python
import torch
from argparse import Namespace
from src.model import Model

args             = Namespace()
args.n_outputs   = 18    # Number of abnormalities
args.embed_dim   = 512   # Dimension of latent space
args.depth       = 1     # Number of spectral blocks
args.hidden_size = [512] # Hidden size in GNN module
args.window_size = [16]  # Receptive field
args.nb_triplets = 80    # Number of triplets of axial slices
args.K           = 3     # Spectral filter size
args.path_resnet = None  # Path of pretrained resnet
args.dropout     = 0.2   # Dropout
args.bias        = True  # Bias
args.spacing_z   = 1.5   # z-axis spacing in mm

device = args.device = torch.device('cpu')

model = Model(args)
model.to(device);

volumes = torch.randn(1, 1, 240, 480, 480).to(device)
labels  = torch.randint(0, 2, (1, 18)).float().to(device)

predictions, loss = model(volumes, labels)
```
