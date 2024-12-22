'''
This is an efficient implementation of the Prim's algorithm to find the minimum spanning tree (MST) that uses Heap data structure. 
The heap implements a min priority queue and stores the node with (current) min cost to be added to the MST in the next iteration.
The running time complexity of this algorithm is O(mlogn); m = # edges, n = #nodes
'''

from functools import reduce
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network


def heapify_up(h, i):
    if(i == 1):
        return
        
    if(h[i][1] < h[i//2][1]):    #// is floor division
        t = h[i//2]
        h[i//2] = h[i]
        h[i] = t
        heapify_up(h, i//2)

def heapify_down(h, i):
    if(2*i >= len(h)):
    # if(2*i >= len(h) - 1):
        return

    if(2*i < len(h) - 1):
        j = 2*i if(h[2*i][1] < h[2*i + 1][1]) else 2*i + 1
        # print('i, j: ', i, j)
    elif(2*i == len(h) - 1):
        j = 2*i

    if(h[i][1] > h[j][1]):
        t = h[j]
        h[j] = h[i]
        h[i] = t
        heapify_down(h, j)        

def extract_min(h):
    m = h[1]
    h[1] = h[len(h) - 1]
    del h[len(h) - 1]
    heapify_down(h,1)
    return m

def delete_heap_node(h, w):
    for i in range(len(h)):
        if(h[i][0] == w):
            break

    deleted_node = h[i]
    #use the index i to delete the node and fix the damaged heap
    h[i] = h[len(h) - 1]
    del h[len(h) - 1]

    # last node was deleted; no need to heapify
    if(len(h) == i):
        return deleted_node
    
    if(i == 1):
        heapify_down(h,i)
    elif(h[i][1] < h[i//2][1]):      #// is floor division
        heapify_up(h,i)
    else:
        heapify_down(h, i)

    return deleted_node     

def insert_heap_node(h, node):
    h.append(node)
    heapify_up(h, len(h) - 1)

def calculate_inf_cost(adj_lst):
    all_edges = reduce(lambda xs, ys: xs + ys, adj_lst.values())
    max_cost = max(list(map(lambda x: x[1], all_edges)))
    return max_cost*2    

def build_heap(adj_lst):
    inf_cost = calculate_inf_cost(adj_lst)
    heap = [(0,0,0)] + [(z[0],z[1],1) for z in sorted(adj_lst[1], key = lambda x: x[1])]
    nodes_not_linked_to_1 = [z for z in list(adj_lst.keys()) if z not in list(map(lambda x: x[0], adj_lst[1])) + [1]]
    heap = heap + list(map(lambda x: (x, inf_cost, 1), nodes_not_linked_to_1))
    return heap

def build_adj_list(input_file):
    adj_lst = {}

    edge_file = open(input_file)

    for line in edge_file:
        edge = line.split(' ')
        edge = [int(e) for e in edge]

        if(len(edge) == 3):
            if(edge[0] in adj_lst):
                adj_lst[edge[0]].append((edge[1], edge[2]))
            else:
                adj_lst[edge[0]] = [(edge[1], edge[2])] 

            if(edge[1] in adj_lst):
                adj_lst[edge[1]].append((edge[0], edge[2]))
            else:
                adj_lst[edge[1]] = [(edge[0], edge[2])]

    return adj_lst    

def build_mst(adj_lst, heap):
    cost = 0
    v = set(list(adj_lst.keys())) #all nodes
    x = {1}  #stores nodes of MST; initial node = 1
    e = []  #stores edges of the MST

    execution_time_begin = datetime.now()
    while (x != v):
        min = extract_min(heap)
        x.add(min[0])
        cost += min[1]
        e.append((min[0], min[2]))

        for w in [z for z in adj_lst[min[0]] if z[0] not in x]:
            w_heap = delete_heap_node(heap, w[0])
            if(w[1] < w_heap[1]):
                # w_heap = w
                w_heap = (w[0], w[1], min[0])
            insert_heap_node(heap, w_heap)

    execution_time_end = datetime.now()

    print('MST cost (with heap): ', cost)
    print('nodes in MST: ', len(x))
    print('edges in MST: ', len(e))

    execution_time = (execution_time_end - execution_time_begin).total_seconds() * 1000
    print(f"The execution time is {execution_time} milliseconds.")

    return e

def plot_mst(e):
    G_ALL_EDGES = nx.DiGraph(e)

    # Add labels to each node
    for node in G_ALL_EDGES.nodes():
        G_ALL_EDGES.nodes[node]['label'] = f'Node {node}' 

    net = Network(notebook=True, cdn_resources='remote')

    net.set_options("""
    var options = {
        "physics": {
            "enabled": true,
            "stabilization": {
                "enabled": true,
                "iterations": 500
            },
            "maxVelocity": 20
        }
    }
    """)

    net.from_nx(G_ALL_EDGES)
    net.show("mst_heap.html")        

if __name__ == "__main__":
    adj_lst = build_adj_list('clustering1.txt')
    heap = build_heap(adj_lst)
    mst_edges = build_mst(adj_lst, heap)           
    plot_mst(mst_edges)