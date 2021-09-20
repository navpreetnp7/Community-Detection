import numpy as np
import torch
import pickle as pkl
import networkx as nx
import pycombo
from sklearn.metrics import normalized_mutual_info_score as nmi

def load_data(daily=False):

    if daily:
        graph = pkl.load(open("data/Taxi2017_daily_net.pkl", "rb"))
        g = [x[0] for x in graph[0]]
        adj = np.array([np.array(nx.adjacency_matrix(x).todense(), dtype=float) for x in g])
        adj = adj[:,1:, 1:]
        return np.array(adj)
    else:
        graph = pkl.load(open("data/Taxi2017_total_net.pkl", "rb"))
        adj = np.array([nx.adjacency_matrix(graph[0]).todense()], dtype=float)
        adj = adj[:,1:, 1:]
        return np.array(adj)

def toy_data():

    graph = nx.DiGraph()
    graph.add_nodes_from([1, 2, 3, 4, 5])
    graph.add_edge(1, 2, weight=10)
    graph.add_edge(1, 5, weight=57)
    graph.add_edge(2, 1, weight=8)
    graph.add_edge(2, 4, weight=34)
    graph.add_edge(2, 5, weight=75)
    graph.add_edge(4, 1, weight=24)
    graph.add_edge(5, 4, weight=14)
    graph.add_edge(5, 1, weight=73)
    graph.add_edge(5, 2, weight=48)

    adj = np.array([nx.adjacency_matrix(graph).todense()], dtype=float)

    return adj

def nmi_score(adj1,adj2):

    G1 = nx.from_numpy_matrix(np.array(adj1))
    G2 = nx.from_numpy_matrix(np.array(adj2))
    partition1 = pycombo.execute(G1)
    partition2 = pycombo.execute(G2)
    return nmi(list(partition1[0]),list(partition2[0]))