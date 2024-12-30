'''
Optimized implementation of Kruskal's algorithm that uses Union-Find data structure.
The running time complexity of this algorithm is O(mlogm); m = # edges.
For a graph with 500 nodes and ~125,000 edges this optimized (union-find) algorithm takes 25 milliseconds to execute and the standard algorithm 865 milliseconds.
Plotting the graph shows that the minimum spanning tree generated by this code is indeed a tree as there are no cycles in the plotted graph.
'''

from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

class UnionFind:
    def __init__(self, nodes):
        self.__uf_sets = {node: [node] for node in nodes}
        self.__uf_ds = {node: node for node in nodes}

    def find(self, u):
        return self.__uf_ds[u]

    def union(self, a, b):
        if(len(self.__uf_sets[self.__uf_ds[a]]) >= len(self.__uf_sets[self.__uf_ds[b]])):
            self.__uf_sets[self.__uf_ds[a]] = self.__uf_sets[self.__uf_ds[a]] + self.__uf_sets[self.__uf_ds[b]]
            set_to_be_deleted = self.__uf_ds[b]
            for e in self.__uf_sets[self.__uf_ds[b]]:
                self.__uf_ds[e] = self.__uf_ds[a]
            del self.__uf_sets[set_to_be_deleted]
        else:
            self.__uf_sets[self.__uf_ds[b]] = self.__uf_sets[self.__uf_ds[b]] + self.__uf_sets[self.__uf_ds[a]]
            set_to_be_deleted = self.__uf_ds[a]
            for e in self.__uf_sets[self.__uf_ds[a]]:
                self.__uf_ds[e] = self.__uf_ds[b]            
            del self.__uf_sets[set_to_be_deleted]

def read_input_file(filename):  
    edges = []
    nodes = set()
    
    lines = open(filename)

    for line in lines:
        tokens = [int(item) for item in line.split(' ')]
        if(len(tokens) == 3):
            edges.append((tokens[0], tokens[1], tokens[2]))
            nodes.add(tokens[0])
            nodes.add(tokens[1])

    return (nodes, edges)

def plot_mst(e):
    G = nx.DiGraph(e)

    # Add labels to each node
    for node in G.nodes():
        G.nodes[node]['label'] = f'Node {node}' 

    net = Network(notebook=True, cdn_resources='remote')

    net.set_options("""
    var options = {
        "physics": {
            "enabled": true,
            "stabilization": {
                "enabled": true,
                "iterations": 500
            },
            "maxVelocity": 5
        }
    }
    """)

    net.from_nx(G)
    net.show("mst_kruskal_union_find.html")

def run_kruskals_uf(v, edges, uf):
    current_time_begin = datetime.now()
    edges = sorted(edges, key=lambda x: x[2])
    x = set()
    mst_edges = []
    mst_cost = 0

    for edge in edges:
        if(x == v):
            break

        if(uf.find(edge[0]) != uf.find(edge[1])):
            uf.union(edge[0], edge[1])
            x.add(edge[0])
            x.add(edge[1])
            mst_edges.append((edge[0], edge[1]))
            mst_cost += edge[2]

    current_time_end = datetime.now()            

    print('mst_cost: ', mst_cost)
    print('# MST edges', len(mst_edges))
    plot_mst(mst_edges)

    execution_time = (current_time_end - current_time_begin).total_seconds() * 1000
    print(f"The execution time is {execution_time} milliseconds.")        

def make_union_find(nodes):
    return {node: node for node in nodes}

if __name__ == '__main__':
    nodes, edges = read_input_file('clustering1.txt')
    union_find_ds = UnionFind(nodes)
    run_kruskals_uf(nodes, edges, union_find_ds)