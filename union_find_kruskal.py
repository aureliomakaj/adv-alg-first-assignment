from distutils.command.check import HAS_DOCUTILS
import math
from os.path import join
from platform import node
from time import perf_counter_ns
from random import random, seed
from random import randint
import gc

#import matplotlib.pyplot as plt

graphs_dir = "mst_dataset"        

class UnionFind:
    def __init__(self, nodes) -> None:
        self.ufset = {}
        self.sizes = {}
        self.initialize(nodes)

    def initialize(self, nodes):
        for node in nodes:
            self.ufset[node] = node
            self.sizes[node] = 1

    def find(self, x):
        parent = self.ufset[x]
        if x == parent:
            return x

        return self.find(parent)

    def size(self, x):
        return self.sizes[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if self.size(root_x) >= self.size(root_y):
            self.ufset[root_y] = root_x
            self.sizes[root_x] += self.size(root_y)
        else:
            self.ufset[root_x] = root_y
            self.sizes[root_y] += self.size(root_x)

def kruskal_union_find(graph, edges):
    """
        Kruskal implementation for MST. 
    """
    res = []
    union_find = UnionFind(graph.keys())
    #Sort edges by weight
    sorted_edges_by_w = sorted(edges, key=lambda tup: tup[2])
    #Iterate edges in nor-decreasing order
    for tuple in sorted_edges_by_w:
        n1, n2, w = tuple
        if union_find.find(n1) != union_find.find(n2):
            res.append(tuple)
            union_find.union(n1, n2)

    return res

def measure_run_time(graph, edges, num_calls, num_instances):
    sum_times = 0.0
    for i in range(num_instances):
        gc.disable() #Disable garbage collector
        start_time = perf_counter_ns() 
        for i in range(num_calls):
            kruskal_union_find(graph, edges)
        end_time = perf_counter_ns()
        gc.enable()
        sum_times += (end_time - start_time)/num_calls
    avg_time = int(round(sum_times/num_instances))
    # return average time in nanoseconds
    return avg_time

def measure_graphs_times(graphs, edges_map):
    num_calls = 10
    num_instances = 10
    run_times = [measure_run_time(graphs[element]['graph'], edges_map[element], num_calls, num_instances) for element in graphs]
    ratios = [None] + [round(run_times[i+1]/run_times[i],3) for i in range(len(graphs.keys())-1)]
    asympt = []
    for i in range(len(graphs.keys())):
        asympt.append(round(graphs[i]['edges'] * math.log2(graphs[i]['nodes']))) 

    size_ratios = [None]
    for i in range(len(asympt)-1):
        size_ratios.append(round(asympt[i+1] /asympt[i], 3))
        
    #c_estimates = [round(run_times[i]/graphs[i],3) for i in range(len(graphs))]
    print("Nodes\tEdges\tAsym\tSR\tTime(ns)\tRatio")
    print(50*"-")
    for i in graphs:
        print(graphs[i]['nodes'], graphs[i]['edges'], asympt[i], size_ratios[i],run_times[i], ratios[i], sep="\t")
    print(50*"-")

def print_mst_graphs_weight(graphs, edges):
    for index in graphs:
        res = kruskal_union_find(graphs[index]['graph'], edges[index])
        sum = 0
        for elem in res:
            sum += elem[2]
        print(sum)

if __name__ == "__main__":
    files = [
        "input_random_17_100.txt",
        "input_random_21_200.txt",
        "input_random_25_400.txt",
        "input_random_29_800.txt",
    ]
    
    graphs = {}
    edges_map = {}
    j = 0
    for filename in files:
        file_graph = open(join(graphs_dir, filename))
        nodes, edges = file_graph.readline().split(" ")

        graph = {}
        edges_list= []
        for i in range(int(edges)):
            n1, n2, w = file_graph.readline().strip().split(" ")
            n1 = int(n1)
            n2 = int(n2)
            w = int(w)
            
            if graph.get(n1) == None:
                graph[n1] = []
            if graph.get(n2) == None:
                graph[n2] = []
            
            graph[n1].append((n2, w))
            graph[n2].append((n1, w))
            edges_list.append((n1, n2,  w))

        graphs[j] = {
            'nodes': int(nodes),
            'edges': int(edges),
            'graph': graph
        }

        edges_map[j] = edges_list
        j += 1

    measure_graphs_times(graphs, edges_map)
    #print_mst_graphs_weight(graphs, edges_map)
