from distutils.command.check import HAS_DOCUTILS
from os.path import join
from platform import node
from time import perf_counter_ns
from random import random, seed
from random import randint
import gc
from collections import OrderedDict
import matplotlib.pyplot as plt


#import matplotlib.pyplot as plt

graphs_dir = "mst_dataset"        

def dfs_search_cycle(graph, v, node_map, edges_map):
    """
        Depth-First Search to detect a cycle.
        We use a map for the nodes, to take trace of the nodes visited,
        and a map for the edges, for the same reason.
        Inseat of labelling DISCOVERY_EDGE or BACK_EDGE, we set the value 
        to True when the edge is visited for the first time, while when 
        we reach a node already visited, we return True, since we detected a cycle
    """
    #Set the node as visited
    node_map[v] = 1
    #For each incident edge of node v
    for tuple in graph[v]:
        opposite, w = tuple
        #If the edge is been visited for the first time
        if edges_map.get((v, opposite)) == None and edges_map.get((opposite, v)) == None :
            if node_map.get(opposite) == None:
                #If it is the first time we visit node 'opposite', we keep searching
                edges_map[(v, opposite)] = True
                edges_map[(opposite, v)] = True
                cycle_presence = dfs_search_cycle(graph, opposite, node_map, edges_map)
                if cycle_presence:
                    #If we detected a cycle in deeper searches, return 
                    return cycle_presence
            else:
                #If the node is met twice, we found a cycle
                return True;

    #No cycles were found
    return False


def not_make_cycle(edges, edge):
    """
        Check if adding edge to edge_list does not create a cycle
    """
    #Build a graph with the edges
    graph = {}
    for tuple in (edges + [edge]):
        v1, v2, w = tuple
        if graph.get(v1) == None:
            graph[v1] = []

        if graph.get(v2) == None:
            graph[v2] = []
        
        graph[v1].append((v2, w))
        graph[v2].append((v1, w))

    #Check for cycle
    cycle = dfs_search_cycle(graph, edge[0], {}, {})
    return not cycle

def kruskal_naive(edges):
    """
        Naive Kruskal implementation for MST. 
    """
    res = []
    #Sort edges by weight
    sorted_edges_by_w = sorted(edges, key=lambda tup: tup[2])
    #Iterate edges in nor-decreasing order
    for tuple in sorted_edges_by_w:
        if not_make_cycle(res, tuple):
            #Add edge if not making a cycle
            res.append(tuple)

    return res

def measure_run_time(edges, num_calls, num_instances):
    sum_times = 0.0
    for i in range(num_instances):
        gc.disable() #Disable garbage collector
        start_time = perf_counter_ns() 
        for i in range(num_calls):
            kruskal_naive(edges)
        end_time = perf_counter_ns()
        gc.enable()
        sum_times += (end_time - start_time)/num_calls
    avg_time = int(round(sum_times/num_instances))
    # return average time in nanoseconds
    return avg_time

def measure_graphs_times(graphs, edges_map):
    num_calls = 1
    num_instances = 1
    run_times = [measure_run_time(edges_map[element], num_calls, num_instances) for element in graphs]
    ratios = [None] + [round(run_times[i+1]/run_times[i],3) for i in range(len(graphs.keys())-1)]

    sizes = [graphs[i]['edges'] + graphs[i]['nodes'] for i in range(len(graphs.keys()))]
    size_ratios = [None] + [round(sizes[i+1] /sizes[i], 3) for i in range(len(sizes)-1)]

    c_estimates = [round(run_times[i]/graphs[i]['edges'] + graphs[i]['nodes'],3) for i in range(len(graphs.keys()))]
    
    print("Size\tSR\tEstimates\tTime(ns)\tRatio")
    print(50*"-")
    for i in graphs:
        print(sizes[i], size_ratios[i], c_estimates[i], run_times[i], ratios[i], sep="\t")
    print(50*"-")

    reference = [1400 * graphs[i]['edges'] * math.log2(graphs[i]['nodes']) for i in range(len(graphs.keys()))]
    plt.plot(run_times, sizes)
    #plt.plot(list_sizes, reference)
    plt.legend(["Measured time"])
    plt.xlabel('run time (ns)')
    plt.ylabel('size')
    plt.show()

def print_mst_graphs_weight(graphs, edges):
    for index in graphs:
        edges_list = edges[index]
        res = kruskal_naive(edges_list)
        sum = 0
        for elem in res:
            sum += elem[2]
        print(sum)

if __name__ == "__main__":
    files = [
        "input_random_49_10000.txt",
        #"input_random_53_20000.txt",
        #"input_random_57_40000.txt",
        #"input_random_61_80000.txt",
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
