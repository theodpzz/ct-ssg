import torch
import torch.nn as nn

from argparse import Namespace

class Edges(nn.Module):
    def __init__(
            self, 
            args: Namespace,
        ) -> None:
        super(Edges, self).__init__()  

        # Parameters
        self.window_size  = args.window_size
        self.num_layer    = len(self.window_size)
        self.spacing_z    = args.spacing_z
        self.nb_triplets  = args.nb_triplets

        self._init_edges()


    def _init_edges(
            self,
        ) -> None:
        self.edges_index = []
        self.edges_weight = []

        for l in range(self.num_layer):
            idx, wgt = self._get_edge(self.window_size[l])
            self.edges_index.append(idx)
            self.edges_weight.append(wgt)

    def get_edge_weight(
            self, 
            edge_index: torch.Tensor,
        ) -> torch.Tensor:
        # Distance between 2 nodes
        distance = torch.abs(edge_index[0, :] - edge_index[1, :]) * self.spacing_z

        # Edge weights
        edge_weight = 1 / (1 + distance)

        return edge_weight

    def _get_edge(
            self, 
            receptive_field: int,
        ) -> torch.Tensor:

        # Minimum and maximum indice of slices
        min_ind   = 0
        max_ind   = self.nb_triplets - 1

        # Edge index list
        edge_index = []

        # Iterate over nodes
        for i in range(self.nb_triplets - 1):  # Connect nearest neighbors

            # Add an edge to itself
            edge_index.append([i, i])

            # Iterate over neighbours
            for receptive_field in range(1, receptive_field + 1):

                # First case
                if(i - receptive_field >= min_ind):
                    if([i - receptive_field, i]) not in edge_index: 
                        edge_index.append([i - receptive_field, i])
                    if([i, i - receptive_field]) not in edge_index: 
                        edge_index.append([i, i - receptive_field])

                # Second case
                if(i + receptive_field <= max_ind):
                    if([i, i + receptive_field]) not in edge_index: 
                        edge_index.append([i, i + receptive_field])
                    if([i + receptive_field, i]) not in edge_index: 
                        edge_index.append([i + receptive_field, i])

        # Permute axis
        edge_index = torch.tensor(edge_index).permute(1, 0)

        # Weights initialization
        edge_weight = self.get_edge_weight(edge_index)

        return edge_index, edge_weight
