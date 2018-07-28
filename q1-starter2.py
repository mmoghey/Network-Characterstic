################################################################################
# # Starter code for Problem 1
# Author: praty@stanford.edu
# Last Updated: Sep 28, 2017
################################################################################

import snap
import numpy as np
import random
import matplotlib.pyplot as plt

# Setup
erdosRenyi = None
smallWorld = None
collabNet = None


# Problem 1.1
def genErdosRenyi(N=5242, E=14484):
    """
    :param - N: number of nodes
    :param - E: number of edges

    return type: snap.PUNGraph
    return: Erdos-Renyi graph with N nodes and E edges
    """
    ############################################################################
    # TODO: Your code here!
    Graph = snap.TUNGraph.New()
    
    for i in range (0,N):
        Graph.AddNode(i)
	
    adj = np.zeros ((N,N))
    count = 0

    while (count <= E+1):
	src = random.randint(0,N-1)
	dest = random.randint(0,N-1)
	if (src != dest):
	    if(adj[src][dest] == 0):
		adj[src][dest] = 1
		count = count + 1
		Graph.AddEdge(src,dest)
	
    Count = snap.CntUniqUndirEdges(Graph)
    print Count
    ############################################################################
    return Graph


def genCircle(N=5242):
    """
    :param - N: number of nodes

    return type: snap.PUNGraph
    return: Circle graph with N nodes and N edges. Imagine the nodes form a
        circle and each node is connected to its two direct neighbors.
    """
    ############################################################################
    # TODO: Your code here!
    Graph = snap.TUNGraph.New()

    for i in range (0,N):
        Graph.AddNode(i)

    print Graph.GetNodes()

    for i in range (0,N):
	#print i%N, (i+1)%N, (i+2)%N
	Graph.AddEdge(i%N, (i+1)%N)
	
    Count = snap.CntUniqUndirEdges(Graph)
    print Count

    	

    ############################################################################
    return Graph


def connectNbrOfNbr(Graph, N=5242):
    """
    :param - Graph: snap.PUNGraph object representing a circle graph on N nodes
    :param - N: number of nodes

    return type: snap.PUNGraph
    return: Graph object with additional N edges added by connecting each node
        to the neighbors of its neighbors
    """
    ############################################################################
    print Graph.GetNodes()

    for i in range (0,N):
	#print i%N, (i+1)%N, (i+2)%N
	Graph.AddEdge(i%N, (i+2)%N)
	
    Count = snap.CntUniqUndirEdges(Graph)
    print Count

    ############################################################################
    return Graph


def connectRandomNodes(Graph, M=4000):
    """
    :param - Graph: snap.PUNGraph object representing an undirected graph
    :param - M: number of edges to be added

    return type: snap.PUNGraph
    return: Graph object with additional M edges added by connecting M randomly
        selected pairs of nodes not already connected.
    """
    ############################################################################
    count = 0
    N = Graph.GetNodes()
    adj = np.zeros ((N,N))

    for i in range (0,N):
	for j in range(0,N):
	    if (Graph.IsEdge(i,j)):
		adj[i][j] = 1

    while (1):
	src = random.randint(0,N)
	dest = random.randint(0,N)
	if (src%N != dest%N):
	    if(adj[src%N][dest%N] == 0):
		adj[src%N][dest%N] = 1
		Graph.AddEdge(src%N,dest%N)
		count = count + 1
		if (count == M):
		    break

    Count = snap.CntUniqUndirEdges(Graph)
    print Count

    ############################################################################
    return Graph


def genSmallWorld(N=5242, E=14484):
    """
    :param - N: number of nodes
    :param - E: number of edges

    return type: snap.PUNGraph
    return: Small-World graph with N nodes and E edges
    """
    Graph = genCircle(N)
    Graph = connectNbrOfNbr(Graph, N)
    Graph = connectRandomNodes(Graph, 4000)
    return Graph


def loadCollabNet(path):
    """
    :param - path: path to edge list file

    return type: snap.PUNGraph
    return: Graph loaded from edge list at `path and self edges removed

    Do not forget to remove the self edges!
    """
    ############################################################################
    # TODO: Your code here!
    Graph = snap.LoadEdgeList(snap.PUNGraph, "C:\Users\manas\Documents\eBooks\Advanced Databases\HomeWork2\ca-GrQc\CA-GrQc.txt", 0, 1)
    for NI in Graph.Nodes():
        if (Graph.IsEdge(NI.GetId(), NI.GetId())):
	    Graph.DelEdge(NI.GetId(), NI.GetId())

    ############################################################################
    return Graph


def getDataPointsToPlot(Graph):
    """
    :param - Graph: snap.PUNGraph object representing an undirected graph
    
    return values:
    X: list of degrees
    Y: list of frequencies: Y[i] = fraction of nodes with degree X[i]
    """
    ############################################################################
    #find out the max out degree
    l1 = []
    
    for NI in Graph.Nodes():
        l1.append(NI.GetOutDeg())

    maxOutDegree = max(l1)

    # populate list l2 with the count of nodes with out degree as index ( for eg., l2[outdegree] = count)
    l2 = []

    #allocate the memory first
    for x in range(0, maxOutDegree+1):
        l2.append(snap.CntDegNodes(Graph, x))

    #for NI in Graph.Nodes() :
    #    l2[NI.GetOutDeg()] = l2[NI.GetOutDeg()] + 1

    # populate x and y as np array
    Y = np.array(l2)
    X = np.array(list(range(0,maxOutDegree+1)))
    
    ############################################################################

    #NId1 = snap.GetMxDegNId(Graph)
    #NI = Graph.GetNI(NId1)
    #maxDeg = NI.GetDeg()
 
    #for x in range(0, maxDeg + 1):
    #    l2.append(snap.CntDegNodes(Graph, x))   

    #Y = np.array(l2)
    #X = np.array(list(range(0,maxDeg+1)))
    return X, Y


def Q1_1():
    """
    Code for HW1 Q1.1
    """
    global erdosRenyi, smallWorld, collabNet
    erdosRenyi = genErdosRenyi(5242, 14484)
    smallWorld = genSmallWorld(5242, 14484)
    collabNet = loadCollabNet("ca-GrQc.txt")

    x_erdosRenyi, y_erdosRenyi = getDataPointsToPlot(erdosRenyi)
    plt.loglog(x_erdosRenyi, y_erdosRenyi, color = 'y', label = 'Erdos Renyi Network')

    x_smallWorld, y_smallWorld = getDataPointsToPlot(smallWorld)
    plt.loglog(x_smallWorld, y_smallWorld, linestyle = 'dashed', color = 'r', label = 'Small World Network')

    x_collabNet, y_collabNet = getDataPointsToPlot(collabNet)
    plt.loglog(x_collabNet, y_collabNet, linestyle = 'dotted', color = 'b', label = 'Collaboration Network')

    plt.xlabel('Node Degree (log)')
    plt.ylabel('Proportion of Nodes with a Given Degree (log)')
    plt.title('Degree Distribution of Erdos Renyi, Small World, and Collaboration Networks')
    plt.legend()
    plt.show()


# Execute code for Q1.1
Q1_1()

# Problem 1.2

# Find max degree of all 3 graphs for plotting (add 2 for padding)
maxdeg = max([erdosRenyi.GetNI((snap.GetMxDegNId(erdosRenyi))).GetDeg(),
                smallWorld.GetNI((snap.GetMxDegNId(smallWorld))).GetDeg(),
                collabNet.GetNI((snap.GetMxDegNId(collabNet))).GetDeg()]) + 2

# Erdos Renyi
def calcQk(Graph, maxDeg=maxdeg):
    """
    :param Graph - snap.PUNGraph object representing an undirected graph
    :param maxDeg - maximum degree(+1) for which q_k needs to be calculated
    
    return type: np.array
    return: array q_k of dimension maxDeg representing the excess degree
        distribution  
    """
    ############################################################################
    # TODO: Your code here!
    q_k = np.zeros(maxDeg)
    for deg in range (0, maxDeg):
        count = 0
        for v in Graph.Nodes():
	    for nid in v.GetOutEdges():
                NI = Graph.GetNI(nid)
		if (NI.GetOutDeg() == deg+1):
                    count = count + 1
	q_k[deg] = count
    
    sum = 0
    for deg in range (0, maxDeg):
        sum = sum + q_k[deg]

    for deg in range (0, maxDeg):
        q_k[deg] = q_k[deg]/sum   

    ############################################################################
    return q_k


def calcExpectedDegree(Graph):
    """
    :param Graph - snap.PUNGraph object representing an undirected graph

    return type: float
    return: expected degree of Graph
    """
    ############################################################################
    # TODO: Your code here!
    ed = 0.0
    
    #find out the max out degree
    l1 = []

    cnt = 0
    for NI in Graph.Nodes():
        cnt = cnt + 1
        l1.append(NI.GetOutDeg())
      

    maxOutDegree = max(l1)

    # populate list l2 with the count of nodes with out degree as index ( for eg., l2[outdegree] = count)
    l2 = []
    #allocate the memory first
    for x in range(0, maxOutDegree+1):
        l2.append(0)

    for NI in Graph.Nodes() :
        l2[NI.GetOutDeg()] = l2[NI.GetOutDeg()] + 1

    for x in range(1, maxOutDegree+1):
        ed = ed + (1.0 * (x * l2[x]) / cnt)

    ###########################################################################        
    return ed


def calcExpectedExcessDegree(Graph, qk):
    """
    :param Graph - snap.PUNGraph object representing an undirected graph
    :param qk - np.array of dimension maxdeg representing excess degree
        distribution of `Graph

    return type: float
    return: expected excess degree of `Graph
    """
    ############################################################################
    # TODO: Your code here!
    eed = 0.0
    i = 0
    for x in qk:
        eed = eed + x * i
        i = i + 1
	#print x, qk
   

    ############################################################################
    return eed


def Q1_2_a():
    """
    Code for Q1.2a
    """
    qk_erdosRenyi = calcQk(erdosRenyi, maxdeg)
    qk_smallWorld = calcQk(smallWorld, maxdeg)
    qk_collabNet = calcQk(collabNet, maxdeg)

    plt.loglog(range(maxdeg), qk_erdosRenyi, color = 'y', label = 'Erdos Renyi Network')
    plt.loglog(range(maxdeg), qk_smallWorld, linestyle = 'dashed', color = 'r', label = 'Small World Network')
    plt.loglog(range(maxdeg), qk_collabNet, linestyle = 'dotted', color = 'b', label = 'Collaboration Network')

    plt.xlabel('k Degree')
    plt.ylabel('Excess Degree Distribution')
    plt.title('Excess Degree Distribution of Erdos Renyi, Small World, and Collaboration Networks')
    plt.legend()
    plt.show()

    # Calculate Expected Degree
    ed_erdosRenyi = calcExpectedDegree(erdosRenyi)
    ed_smallWorld = calcExpectedDegree(smallWorld)
    ed_collabNet = calcExpectedDegree(collabNet)
    print 'Expected Degree for Erdos Renyi: %f' % ed_erdosRenyi
    print 'Expected Degree for Small World: %f' % ed_smallWorld
    print 'Expected Degree for Collaboration Network: %f' % ed_collabNet

    # Calculate Expected Excess Degree
    eed_erdosRenyi = calcExpectedExcessDegree(erdosRenyi, qk_erdosRenyi)
    eed_smallWorld = calcExpectedExcessDegree(smallWorld, qk_smallWorld)
    eed_collabNet = calcExpectedExcessDegree(collabNet, qk_collabNet)
    print 'Expected Excess Degree for Erdos Renyi: %f' % (eed_erdosRenyi)
    print 'Expected Excess Degree for Small World: %f' % (eed_smallWorld)
    print 'Expected Excess Degree for Collaboration Network: %f' % (eed_collabNet)


# Execute code for Q1.2a
Q1_2_a()


# Problem 1.3 - Clustering Coefficient

def calcClusteringCoefficient(Graph):
    """
    :param - Graph: snap.PUNGraph object representing an undirected graph

    return type: float
    returns: clustering coeffient of `Graph 
    """    
    ############################################################################
    # TODO: Your code here!
    C = 0.0
    ei = 0
    V = 0
    for NI in Graph.Nodes():
	for nid1 in NI.GetOutEdges():
	    for nid2 in NI.GetOutEdges():
                if (nid1 != nid2):
		    if (Graph.IsEdge(nid1, nid2)):
                        ei = ei + 1
	ki = NI.GetDeg()
	ei = ei / 2 
	if (ki >= 2):
	    Ci =  (2 * ei) / ((1.0) * (ki * (ki - 1)))
	    #print ei, Ci
	else:
	    Ci = 0
        C = C + Ci
	V = V + 1
    print C, V
    C = C /(1.0 * V)
                    
    ############################################################################
    return C

def Q1_3():
    """
    Code for Q1.3
    """
    C_erdosRenyi = calcClusteringCoefficient(erdosRenyi)
    C_smallWorld = calcClusteringCoefficient(smallWorld)
    C_collabNet = calcClusteringCoefficient(collabNet)
    
    print('Clustering Coefficient for Erdos Renyi Network: %f' % C_erdosRenyi)
    print('Clustering Coefficient for Small World Network: %f' % C_smallWorld)
    print('Clustering Coefficient for Collaboration Network: %f' % C_collabNet)

    CfVec = snap.TFltPrV()
    C_erdosRenyi = snap.GetClustCf(erdosRenyi, CfVec, -1)
    CfVec = snap.TFltPrV()
    C_smallWorld = snap.GetClustCf(smallWorld, CfVec, -1)
    CfVec = snap.TFltPrV()
    C_collabNet = snap.GetClustCf(collabNet, CfVec, -1)
    
    print('Clustering Coefficient for Erdos Renyi Network: %f' % C_erdosRenyi)
    print('Clustering Coefficient for Small World Network: %f' % C_smallWorld)
    print('Clustering Coefficient for Collaboration Network: %f' % C_collabNet)



# Execute code for Q1.3
Q1_3()
