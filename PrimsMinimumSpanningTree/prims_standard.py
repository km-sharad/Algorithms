'''
This is a straight-forward implementation of the Prim's algorithm to find the minimum spanning tree (MST). 
The running time complexity of this algorithm is O(mn); m = # edges, n = #nodes
For a graph with 500 nodes and ~125,000 edges this standard algorithm takes ~60 seconds to execute and the optimized (heap) algorithm ~2 seconds.
'''

from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

def build_adj_list(input_file):    
    adj_lst = {}
    v = set()
    e = []

    edge_file = open(input_file)

    for line in edge_file:
        edge = line.split(' ')
        edge = [int(e) for e in edge]

        if(len(edge) == 3):
            e.append(tuple(edge))
            v.add(edge[0])
            v.add(edge[1])

            if(edge[0] in adj_lst):
                adj_lst[edge[0]].append((edge[0], edge[1], edge[2]))
            else:
                adj_lst[edge[0]] = [(edge[0], edge[1], edge[2])] 

            if(edge[1] in adj_lst):
                adj_lst[edge[1]].append((edge[1], edge[0], edge[2]))
            else:
                adj_lst[edge[1]] = [(edge[1], edge[0], edge[2])]  
    return adj_lst   

def build_mst(adj_lst):
    v = set(list(adj_lst.keys())) # all nodes
    t = [] #stores edges of the MST
    x = {1} #stores nodes; initialized with node 1
    total_cost = 0

    current_time_begin = datetime.now()
    while(x != v):
        edges = []
        for node in x:
            edges = edges + adj_lst[node]

        edges = [z for z in edges if z[1] not in x]
        edges = sorted(edges, key= lambda x: x[2])
        selected_edge = edges[0]
        x.add(selected_edge[1])
        t.append((selected_edge[0], selected_edge[1]))
        total_cost += selected_edge[2]

    current_time_end = datetime.now()

    print('MST cost: ', total_cost)
    print('# nodes in MST',len(x))
    print('# edges in MST',len(t))

    execution_time = (current_time_end - current_time_begin).total_seconds() * 1000
    print(f"The execution time is {execution_time} milliseconds.")

    return t

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
    net.show("mst_standard.html")

if __name__ == '__main__':
    adj_lst = build_adj_list('clustering1.txt')
    mst_edges = build_mst(adj_lst)           
    plot_mst(mst_edges)    
