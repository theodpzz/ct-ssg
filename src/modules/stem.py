import torch
import torch.nn as nn

from torchvision import models
from argparse import Namespace

class SlicesEmbeddings(nn.Module):
    def __init__(
            self, 
            args: Namespace,
        ) -> None:
        super(SlicesEmbeddings, self).__init__()  

        # Parameters
        device     = args.device
        drop_p     = args.dropout
        self.embed_dim   = args.embed_dim
        self.nb_triplets = args.nb_triplets

        # ResNet initialization
        resnet = models.resnet18(weights=None)

        # Load ResNet features
        if(args.path_resnet is not None): 
            resnet.load_state_dict(torch.load(args.path_resnet))
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        
        # Axial positional embeddings
        self._init_position_embeddings(device)

        # dropout
        self.dropout = nn.Dropout(drop_p)

    def _init_position_embeddings(
            self, 
            device: torch.device,
        ) -> None:
        self.position_embeddings = nn.Parameter(torch.randn(1, self.nb_triplets, self.embed_dim)).to(device)
        self.position_embeddings = nn.init.trunc_normal_(self.position_embeddings, std=.02)

    def forward(
            self, 
            volumes: torch.Tensor,
        ) -> torch.Tensor:

        # Extract batch size
        batch_size = volumes.size(0)                                  # batch_size

        # Reshape by grouping channels 3 by 3
        x = volumes.reshape(batch_size*self.nb_triplets, 3, 480, 480) # [batch_size, n_features, 3, H, W]

        # Extract feature maps for each triplet
        x = self.features(x)                                          # [batch_size n_features, dim, h, w]

        # Global Average Pooling from ResNet feature maps
        x = x.mean(dim=(2, 3))                                        # [batch_size n_features, dim]
        
        # Reshape
        x = x.view(batch_size, self.nb_triplets, self.embed_dim)      # [batch_size, n_features, dim]

        # Axial positional embeddings
        x = x + self.position_embeddings                              # [batch_size, n_features, dim]

        # Dropout
        x = self.dropout(x)                                           # [batch_size, n_features, dim]

        return x
