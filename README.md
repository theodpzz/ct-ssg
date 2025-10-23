### CT-SSG 🩺👨🏻‍⚕️

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
args.n_outputs   = 18
args.embed_dim   = 512
args.depth       = 1
args.hidden_size = [512]
args.window_size = [16]
args.nb_triplets = 80
args.K           = 3
args.path_resnet = None
args.dropout     = 0.2
args.bias        = True
args.spacing_z   = 0.015

device = args.device = torch.device('cpu')

model = Model(args)
model.to(device);

volumes = torch.randn(1, 1, 240, 480, 480).to(device)
labels  = torch.randint(0, 2, (1, 18)).float().to(device)

predictions, loss = model(volumes, labels)
```
