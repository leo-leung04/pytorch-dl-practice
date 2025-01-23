# Build up a Graph Neural Network using PyTorch Geometric for Node Property Prediction

#############
# Import packages
#############

import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F

# The PyG built-in GCNConv
from torch_geometric.nn import GCNConv

import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

#############
# Load and Preprocess the Dataset
#############

if 'IS_GRADESCOPE_ENV' not in os.environ:
  dataset_name = 'ogbn-arxiv'
  dataset = PygNodePropPredDataset(name=dataset_name,
                                  transform=T.ToSparseTensor())
  data = dataset[0]

  # Make the adjacency matrix to symmetric
  data.adj_t = data.adj_t.to_symmetric()

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # If you use GPU, the device should be cuda
  print('Device: {}'.format(device))

  data = data.to(device)
  split_idx = dataset.get_idx_split()
  train_idx = split_idx['train'].to(device)

  #############
  # GCN Model
  #############

  class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, return_embeds=False):
        super(GCN, self).__init__()
        ############# Your code here ############
        ## Note:
        ## 1. You should use torch.nn.ModuleList for self.convs and self.bns
        ## 2. self.convs has num_layers GCNConv layers
        ## 3. self.bns has num_layers - 1 BatchNorm1d layers
        ## 4. You should use torch.nn.LogSoftmax for self.softmax
        ## 5. The parameters you can set for GCNConv include 'in_channels' and 
        ## 'out_channels'. More information please refer to the documentation:
        ## https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
        ## 6. The only parameter you need to set for BatchNorm1d is 'num_features'
        ## More information please refer to the documentation: 
        ## https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
        ## (~10 lines of code)

        self.num_layers = num_layers

        # A list of GCNConv layers
        self.convs = nn.ModuleList([GCNConv(input_dim, hidden_dim)])
        self.convs.extends([GCNConv(hidden_dim, hidden_dim) for i in range(num_layers-2)])
        self.convs.extends([GCNConv(hidden_dim, output_dim)])

        # A list of 1D batch normalization layers
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for i in range(num_layers-1)])

        # The log softmax layer
        self.softmax = nn.LogSoftmax(dim=1)

        # Probability of an element to be zeroed
        self.dropout = dropout

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        
    def forward(self, x, adj_t):
        # TODO: Implement this function that takes the feature tensor x,
        # edge_index tensor adj_t and returns the output tensor as
        # shown in the figure.

        ############# Your code here ############
        ## Note:
        ## 1. Construct the network as showing in the figure
        ## 2. torch.nn.functional.relu and torch.nn.functional.dropout are useful
        ## More information please refer to the documentation:
        ## https://pytorch.org/docs/stable/nn.functional.html
        ## 3. Don't forget to set F.dropout training to self.training
        ## 4. If return_embeds is True, then skip the last softmax layer
        ## (~7 lines of code)
        for i in range(self.num_layers):
           
           # Last Conv layer pass
           if (i == self.num_layers-1):
              x = self.convs[i](x, adj_t)
              if (self.return_embeds):
                 return x
              x = self.softmax(x)
              out = x
           else:
              x = self.conv[i](x, adj_t)
              x = self.bns[i](x)
              x = F.relu(x)
              x = F.dropout(x, p=self.dropout, training=self.traning)
        ##########################################
        return out
