import torch
import torch.nn as nn
import torch_geometric as pyg

from argparse import Namespace
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import ChebConv, global_mean_pool

from src.modules.edges import Edges

class OperatorBlocks(nn.Module):
    def __init__(
            self, 
            args: Namespace,
        ) -> None:
        super(OperatorBlocks, self).__init__()

        # Initialize main parameters
        self._build_config(args)

        # Build first block operator
        self._build_blocks()

        # Global mean pooling
        self.pool = global_mean_pool

        # Edges
        self.edges = Edges(args)

    def _build_config(
            self,
            args: Namespace,
    ) -> None:
        # Parameters
        self.device   = args.device

        # Dimension of inputs features
        self.in_channels = args.embed_dim

        # Dimension of hidden features
        self.hidden_sizes = args.hidden_size

        # Dimension of outputs features
        self.out_channels = args.embed_dim

        # Depth of the network: number of layers
        self.num_layers = len(args.hidden_size)

        # Bias and dropout
        self.bias    = args.bias
        self.dropout = args.dropout

        # Spectral filter size
        self.K = args.K

    def _build_blocks(
        self,
    ) -> None:
        
        # Block operator layers
        self.graph_convs = []
        self.graph_convs.append(self._get_convolution(self.in_channels, self.hidden_sizes[0], self.bias, self.K))

        # Normalization layers
        self.lns_in  = []
        self.lns_out = []
        self.lns_in.append(pyg.nn.norm.LayerNorm(self.hidden_sizes[0], affine=False))
        self.lns_out.append(pyg.nn.norm.LayerNorm(self.hidden_sizes[0], affine=False))

        # Feedforward Neural Network
        self.mlp = []
        self.mlp.append(nn.Sequential(
            nn.Linear(self.hidden_sizes[0], self.hidden_sizes[0]), 
            nn.GELU(), 
            nn.Dropout(self.dropout)))    
    
        # Iterate to construct remaining layers
        for l in range(1, self.num_layers):
            self.graph_convs.append(self._get_convolution(self.hidden_sizes[l-1], self.hidden_sizes[l], self.bias, self.K))
            self.lns_in.append(pyg.nn.norm.LayerNorm(self.hidden_sizes[l-1], affine=False))
            self.lns_out.append(pyg.nn.norm.LayerNorm(self.hidden_sizes[l], affine=False))
            self.mlp.append(nn.Sequential(nn.Linear(self.hidden_sizes[l], self.hidden_sizes[l]), nn.GELU(), nn.Dropout(self.dropout)))

        # List to nn.ModuleList
        self.graph_convs = nn.ModuleList(self.graph_convs)
        self.lns_in      = nn.ModuleList(self.lns_in)
        self.lns_out     = nn.ModuleList(self.lns_out)
        self.mlp         = nn.ModuleList(self.mlp)

    def _get_convolution(self, dim_in, dim_out, bias, K):
        operator = ChebConv(dim_in, dim_out, bias=bias, K=K)
        return operator

    def get_batch_format(self, l, node_feature):
        # Batch size
        batch_size = len(node_feature) # batch_size

        # Create Data object for each sample
        graphs = []
        for i in range(batch_size):
            graphs.append(
                Data(
                    x           = node_feature[i], 
                    edge_index  = self.edges.edges_index[l], 
                    edge_weight = self.edges.edges_weight[l],
                    )
                )

        # Batch the graphs
        batch = Batch.from_data_list(graphs)

        return batch
    
    def forward_operator(self, batch_data, l):
        # Forward message passing
        h = self.graph_convs[l](
            x           = batch_data.x,
            edge_index  = batch_data.edge_index.to(self.device),
            edge_weight = batch_data.edge_weight.to(self.device),
        )

        return h
    
    def forward(
            self, 
            node_feature: torch.Tensor,
        ) -> torch.Tensor:

        h = node_feature.clone()

        # Iterate over operator layers
        for l in range(self.num_layers):

            # Residual connection
            h_res = h

            # First normalization layer
            h = self.lns_in[l](x=h)                                 # [batch_size, n_features, dim]

            # PyG Batch Data format
            batch_data = self.get_batch_format(l=l, node_feature=h) # PyG Batch Data

            # Operator forward pass
            h = self.forward_operator(batch_data, l)                # PyG Batch Data

            # PyG Batch Data format to torch tensor
            h, _ = to_dense_batch(h, batch=batch_data.batch)        # [batch_size, n_features, dim]

            # Residual connection
            h = h + h_res                                           # [batch_size, n_features, dim]

            # Residual connection
            h_res = h

            # Nprmalization Layer
            h = self.lns_out[l](x=h)                                # [batch_size, n_features, dim]

            # MLP
            h = self.mlp[l](h)                                      # [batch_size, n_features, dim]

            # Residual connection
            h = h + h_res                                           # [batch_size, n_features, dim]

        # Pooling
        h = torch.mean(h, dim=1)                                    # [batch_size, dim]

        return h
