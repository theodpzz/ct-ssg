import torch
import torch.nn as nn

from argparse import Namespace

from src.modules.stem import SlicesEmbeddings
from src.modules.operator import OperatorBlocks

class Model(nn.Module):
    def __init__(
            self, 
            args = Namespace,
        ) -> None:
        super(Model, self).__init__()  

        # Parameters
        n_outputs = args.n_outputs

        # Slices embedding module
        self.stem = SlicesEmbeddings(args)
        
        # Operator blocks
        self.blocks = OperatorBlocks(args)

        # Classification head
        d = args.hidden_size[-1]
        self.classifier = nn.Sequential(
            nn.Linear(d, d // 2), 
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(d // 2, n_outputs))
        
        # Loss function
        self.loss = nn.BCEWithLogitsLoss(reduction = 'none')

        # Activation function
        self.sigmoid = nn.Sigmoid()

    def getloss(
            self, 
            prediction: torch.Tensor,
            target: torch.Tensor,
        ) -> torch.Tensor:
        loss = self.loss(prediction, target)
        return loss  

    def forward(
            self, 
            volumes: torch.Tensor, 
            labels: torch.Tensor,
        ) -> torch.Tensor:

        # triplet axial slices volume
        x = self.stem(volumes)

        # Operator blocks message passing
        x = self.blocks(x)

        # Classification head to obtain logits
        logits = self.classifier(x)

        # compute loss
        loss = self.getloss(logits, labels)

        # logits to probablities
        predictions = self.sigmoid(logits)

        return predictions, loss
