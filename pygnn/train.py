from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.optim as optim

from utils import load_data,normalize,toy_data,nmi_score,modularity_matrix,modularity
from models import GNN

import community as community_louvain
from networkx import from_numpy_matrix

torch.set_printoptions(sci_mode=False)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=426, help='Random seed.')
parser.add_argument('--epochs', type=int, default=20000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.00001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=10e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--ndim', type=int, default=2,
                    help='Embeddings dimension.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj = load_data(daily=False)
#adj = toy_data()
G = from_numpy_matrix(adj[0])

adj_norm = normalize(adj)

adj = torch.FloatTensor(np.array(adj))
adj_norm = torch.FloatTensor(np.array(adj_norm))

# features
partition = community_louvain.best_partition(G)

#get binary matrix of the partition
nb_community = max(list(partition.values())) + 1
communities =  np.array(list(partition.values())).reshape(-1)
C = np.eye(nb_community)[communities]
features = torch.FloatTensor(C)
features = features.unsqueeze(0)

Q = modularity_matrix(adj)

# Model and optimizer

model = GNN(batch_size=adj.shape[0],
            nfeat=adj.shape[1],
            ndim=args.ndim)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    adj_norm = adj_norm.cuda()
    Q = Q.cuda()

# Train model
t_total = time.time()

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


for epoch in range(args.epochs):

    t = time.time()
    model.train()
    optimizer.zero_grad()

    C = model(features, adj_norm)
    #print(C)
    loss = modularity(C,Q)

    loss.backward()

    optimizer.step()

    if epoch == 0:
        best_loss = loss
    else:
        if loss < best_loss:
            best_loss = loss

    if epoch % 100 == 0:
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss: {:.8f}'.format(best_loss.item()),
              'time: {:.4f}s'.format(time.time() - t))



print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
