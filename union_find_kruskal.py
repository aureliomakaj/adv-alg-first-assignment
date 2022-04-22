import math
from os.path import join
from time import perf_counter_ns
import gc
import matplotlib.pyplot as plt

graphs_dir = "mst_dataset"        

class UnionFind:
    """
        UnionFind data structure for Kruskal's Algorithm. 
        For the sets we use a map where the key is the node and the value is the parent.
        Another map is used to keep trace of the size of each set
    """
    def __init__(self, nodes) -> None:
        self.ufset = {}
        self.sizes = {}
        self.initialize(nodes)

    def initialize(self, nodes):
        """
            Initialization.
            For each node create a singleton set
        """
        for node in nodes:
            self.ufset[node] = node
            self.sizes[node] = 1

    def find(self, x):
        """
            Get the set name of a given element. 
            The name of the set is the name of the root, 
            that is the node whose parent is itself
        """
        parent = self.ufset[x]
        if x == parent:
            return x

        return self.find(parent)

    def size(self, x):
        """
            Get the size of the set of x
        """
        return self.sizes[x]

    def union(self, x, y):
        """
            Union operation between two sets. 
            The root of the smaller set become an internal node
            and points to the root of the bigger set. The size is 
            increased accordingly. 
        """
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
    #Iterate edges in non-decreasing order
    for tuple in sorted_edges_by_w:
        n1, n2, w = tuple
        #If the nodes are in different sets, we can add the 
        #edge since it does not create a cycle
        if union_find.find(n1) != union_find.find(n2):
            res.append(tuple)
            union_find.union(n1, n2)

    return res

def measure_run_time(graph, edges, num_calls, num_instances):
    sum_times = 0.0
    res = None
    for i in range(num_instances):
        gc.disable() #Disable garbage collector
        start_time = perf_counter_ns() 
        for i in range(num_calls):
            res = kruskal_union_find(graph, edges)
        end_time = perf_counter_ns()
        gc.enable()
        sum_times += (end_time - start_time)/num_calls
    avg_time = int(round(sum_times/num_instances))
    print("Finished")
    # return average time in nanoseconds
    return avg_time, res

def measure_graphs_times(graphs, edges_map):
    num_calls = 100
    num_instances = 10
    
    #Compute the avarage time of Prim's algorithm execution on each graph
    mst_results = []
    run_times = []
    for element in graphs:
        time, res = measure_run_time(graphs[element]['graph'], edges_map[element], num_calls, num_instances)
        run_times.append(time)
        mst_results.append(res)

    #Get the ratio between one execution time and the previous
    ratios = [None] + [round(run_times[i+1]/run_times[i],3) for i in range(len(graphs.keys())-1)]

    #Graph size
    sizes = [graphs[i]['edges'] + graphs[i]['nodes'] for i in range(len(graphs.keys()))]
    #Graph size ratio
    size_ratios = [None] + [round(sizes[i+1] /sizes[i], 3) for i in range(len(sizes)-1)]

    #Estimated time
    c_estimates = [round(run_times[i]/(graphs[i]['edges'] * math.log2(graphs[i]['nodes'])),3) for i in range(len(graphs.keys()))]

    print("Nodes\tEdges\tSize\tSR\tEstimates\tTime(ns)\tRatio")
    print(50*"-")
    for i in graphs:
        print(graphs[i]['nodes'], graphs[i]['edges'], sizes[i], size_ratios[i], c_estimates[i], run_times[i], ratios[i], sep="\t")
    print(50*"-")

    for res in mst_results:
        print_mst_graphs_weight(res)

    const_ref = 110
    reference = [const_ref * graphs[i]['edges'] * math.log2(graphs[i]['nodes']) for i in range(len(graphs.keys()))]
    fig, (linear, log) = plt.subplots(2)
    fig.suptitle("Kruskal's algorithm")

    linear.plot(sizes, run_times)
    linear.plot(sizes, reference)
    linear.legend(["Measured time", "Reference (" + str(const_ref) + ")"])

    log.plot(sizes, run_times)
    log.plot(sizes, reference)
    log.legend(["Measured time", "Reference (" + str(const_ref) + ")"])
    log.set_yscale('log')

    plt.ylabel('run time (ns)')
    plt.xlabel('size')
    plt.show()

def print_mst_graphs_weight(res):
    sum = 0
    for elem in res:
        sum += elem[2]
    print(sum)

if __name__ == "__main__":
    files = [
        "input_random_33_1000.txt",
        "input_random_37_2000.txt",
        "input_random_41_4000.txt",
        "input_random_45_8000.txt",
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
