import math
from os.path import join
from time import perf_counter_ns
from random import random, seed
from random import randint
import gc
import matplotlib.pyplot as plt

graphs_dir = "mst_dataset"


class MinHeap:
    """
    Definition of MinHeap Data Structure. It handle some special cases 
    for Prim's algorithm, such as the 'Inf' key value, which is not compatible 
    with native language '<' operation
    """
    def __init__(self, arr) -> None:
        self.heapSize = len(arr)
        self.heap = []
        self.nodes = {}
        #Clone of the array. We don't want to change the original array
        for i in range(self.heapSize):
            self.heap.append(arr[i])
            #Map between a node and its index. 
            #Useful so the operation of checking if a node is in the heap has time O(1)
            self.nodes[arr[i]['node']] = i
        
        #First ordering, starting from the middle down to 1
        for i in reversed(range(self.heapSize // 2 )):
            self.minHeapify(i)


    def parent(self, i):
        return (i - 1) // 2
            
    def left(self, i):
        return (i * 2) + 1

    def right(self, i):
        return (i * 2) + 2

    def isLower(self, first, second):
        """
            Lower operation to handle cases with 'Inf' string value
        """
        if first == 'Inf':
            return False
        elif second == 'Inf':
            return True
        else:
            return first < second
    

    def exchange(self, i, j):
        """
            Exchange element in position i with element in position j. 
        """
        self.nodes[self.heap[i]['node']] = j
        self.nodes[self.heap[j]['node']] = i

        tmp = self.heap[i]
        self.heap[i] = self.heap[j]
        self.heap[j] = tmp

    def minHeapify(self, i):
        """
            Function to maintain the property of the Min Heap, that is that
            the parent is always lower than the left and right child. 
            If it is not, then exchange the values and run again.
        """
        left = self.left(i)
        right = self.right(i)
        #Check if the left child exists and it is not lower then the parent 
        if left < self.heapSize and self.isLower(self.heap[left]['key'], self.heap[i]['key']):
            lowest = left
        else:
            lowest = i

        #Check if the right child exists and is not lower then the lowest between the parent and the left
        if right < self.heapSize and  self.isLower(self.heap[right]['key'], self.heap[lowest]['key']): 
            lowest = right

        #If the lowest is not the parent, fix the order and run again
        if lowest != i:
            self.exchange(i, lowest)
            self.minHeapify(lowest)
    
    def isEmpty(self):
        """
            Check if the Heap is empty
        """
        return self.heapSize == 0

    def hasNode(self, node):
        """
            Check if Heap has a given node
        """
        return self.nodes[node] < self.heapSize

    def minimum(self):
        """
            Get the minimum
        """
        return self.heap[0]

    def extractMin(self):
        """
            Extract the minimum and reorder the Heap
        """
        if self.heapSize < 1:
            print("Heap underflow")
            exit(1)
            
        min = self.heap[0]
        self.exchange(0, self.heapSize - 1)
        self.heapSize -= 1
        self.minHeapify(0)
        
        return min

    def getIndexByNode(self, node):
        """
            Get the index of a given node in the Heap if there is, otherwise -1
        """
        return self.nodes[node] if self.hasNode(node) else -1


    def updateNode(self, node, key, parent):
        """
            Update the data of an element of the heap, 
            and reorder the Heap to keep the MinHeap property true, since 
            the key value can change and become higher then the left or right child
        """
        index = self.getIndexByNode(node)

        if index != -1:
            self.heap[index]['key'] = key
            self.heap[index]['parent'] = parent
            while index > 0 and not self.isLower(self.heap[self.parent(index)]['key'], self.heap[index]['key']):
                self.exchange(self.parent(index), index)
                index = self.parent(index)

def prim(graph, root):
    """
        Prim's algorithm with Heap Implementation for Minimum Spanning Trees.
        Graph is an adjacence list and root is the root of the tree
    """
    supp = {}
    #All the nodes has Inf weight and no parent
    for node in graph.keys():
        supp[node] = {
            'node': node,
            'key': 'Inf',
            'parent': None
        }

    #Update root key to 0. The root will be the first to be extracted
    supp[root]['key'] = 0

    #Create MinHeap based on key value
    q = MinHeap(list(supp.values()))
    while not q.isEmpty():
        #Extract the node with minimum weight
        minimum = q.extractMin()
        #For each node adjacent to minimum node
        for tuple in graph[minimum['node']]:
            v, weight = tuple
            #If the node has not been extracted yet, and the weight of the edge is
            #lower then node's current key, update the key and its parent
            if q.hasNode(v) and q.isLower(weight, supp[v]['key']):
                q.updateNode(v, weight, minimum['node'])
    return supp


def measure_run_time(graph, num_calls, num_instances):
    """
        Execute Prim's algorithm (num_instances * num_calls) times, 
        and get the avarage time in nanoseconds. 
    """
    sum_times = 0.0
    for i in range(num_instances):
        gc.disable() #Disable garbage collector
        start_time = perf_counter_ns() 
        for i in range(num_calls):
            prim(graph, list(graph.keys())[0])
        end_time = perf_counter_ns()
        gc.enable()
        sum_times += (end_time - start_time)/num_calls
    avg_time = int(round(sum_times/num_instances))
    # return average time in nanoseconds
    return avg_time


def measure_graphs_times(graphs):
    num_calls = 1
    num_instances = 1
    
    #Compute the avarage time of Prim's algorithm execution on each graph
    run_times = [measure_run_time(graphs[element]['graph'], num_calls, num_instances) for element in graphs]
    ratios = [None] + [round(run_times[i+1]/run_times[i],3) for i in range(len(graphs.keys())-1)]

    
    sizes = [graphs[i]['edges'] + graphs[i]['nodes'] for i in range(len(graphs.keys()))]
    size_ratios = [None] + [round(sizes[i+1] /sizes[i], 3) for i in range(len(sizes)-1)]

    c_estimates = [round(run_times[i]/(graphs[i]['edges'] * math.log2(graphs[i]['nodes'])),3) for i in range(len(graphs.keys()))]

    print("Nodes\tEdges\tSize\tSR\tEstimates\tTime(ns)\tRatio")
    print(50*"-")
    for i in graphs:
        print(graphs[i]['nodes'], graphs[i]['edges'], sizes[i], size_ratios[i], c_estimates[i], run_times[i], ratios[i], sep="\t")
    print(50*"-")

    reference = [1400 * graphs[i]['edges'] * math.log2(graphs[i]['nodes']) for i in range(len(graphs.keys()))]
    plt.plot(sizes, run_times)
    plt.plot(sizes, reference)
    plt.legend(["Measured time", "Reference"])
    plt.ylabel('run time (ns)')
    #plt.yscale('log')
    plt.xlabel('size')
    plt.show()

def print_mst_graphs_weight(graphs):
    for index in graphs:
        graph = graphs[index]['graph']
        res = prim(graph, list(graph.keys())[0])
        sum = 0
        for index in res:
            sum += res[index]['key']
        print(sum)

if __name__ == "__main__":
    files = [
        "input_random_49_10000.txt",
        "input_random_53_20000.txt",
        "input_random_57_40000.txt",
        "input_random_61_80000.txt",
    ]

     #set of graphs
    graphs = {}
    j = 0
    for filename in files:
        file_graph = open(join(graphs_dir, filename))
        nodes, edges = file_graph.readline().split(" ")

        graph = {}
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

        graphs[j] = {
            'nodes': int(nodes),
            'edges': int(edges),
            'graph': graph
        }
        j += 1

    measure_graphs_times(graphs)
    #print_mst_graphs_weight(graphs)


    