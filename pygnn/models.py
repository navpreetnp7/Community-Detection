import torch.nn as nn
import torch.nn.functional as F
from layers import GraphNeuralNet
import torch


class GNN(nn.Module):

    def __init__(self, batch_size, nfeat, ndim):
        super(GNN, self).__init__()

        self.gc1 = GraphNeuralNet(batch_size, nfeat, ndim)

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = x/x.sum(axis=2).unsqueeze(2) #normalize st sum = 1
        print(x)
        return x
