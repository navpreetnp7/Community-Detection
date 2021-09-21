import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphNeuralNet(Module):

    def __init__(self, batch_size, in_features, out_features):
        super(GraphNeuralNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size

        weight1_eye = torch.FloatTensor(torch.eye(in_features))
        weight1_eye = weight1_eye.reshape((1, in_features, in_features))
        weight1_eye = weight1_eye.repeat(batch_size, 1, 1)
        self.weight1 = Parameter(weight1_eye)
        self.weight2 = Parameter(torch.zeros(batch_size, in_features, in_features))

    def forward(self, input, adj):
        support = self.weight1 + torch.bmm(self.weight2,adj)
        output = torch.bmm(support, input)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'