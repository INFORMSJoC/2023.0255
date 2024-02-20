from gurobipy import *
import numpy as np
import os
import torch
import dgl
from dgl.data.utils import save_graphs
import math

# Numeric labels for each instance class
labelMap = {'vrptw': 0, 'cwlp-ss': 1, 'pcp': 2, 'scheduling': 3, 'cvrp': 4, 'clp': 5, 'inr': 6, 'lotsizing': 7, 'coloring': 8, 'tup': 9, 'bppif': 10, 'maplabeling': 11, 'kps': 12, 'bpp2': 13, 'cuttingstock': 14, 'relaxedClique': 15, 'gap': 16, 'cpmp': 17, 'bpp': 18}

# Load a dictionary where the keys are ILP instance names and the values are the paths to the corresponding lp files (e.g., key: Instance1, value: modelfiles/vrptw/Instance1)
models = {l.strip().split(",")[0]: l.strip().split(",")[1] for l in open('Dictionary_name_here.csv').readlines()}

# Load a dictionary where the keys are ILP instance names and the values are the corresponding class type (e.g., key: Instance1, value: vrptw)
modelLabels = {l.strip().split(",")[0]: labelMap[l.strip().split(",")[2]] for l in open('Dictionary_name_here.csv').readlines()}

# Map Gurobi's naming convention for variable type into integer values
vtypeAsInt = {'B':1, 'I':-1, 'C':0}

# Map Gurobi's naming convention for constraint type into integer values
senseMap = {'=':0, '<':-1, '>':1}

# Create a graph for each instance in *models*
for model in models:
    m = read(models[model])
    m._vars = m.getVars()
    m._cons = m.getConstrs()
    numberOfVars = len(m._vars)
    numberOfCons = len(m._cons)
    print('Model: {}   Vars: {}   Cons: {}'.format(model, numberOfVars, numberOfCons))
    nnodes = numberOfVars + numberOfCons

    # Compile node feature vectors
    firstNode = True
    for i in range(len(m._vars)):
        thisType = vtypeAsInt[m._vars[i].vtype]
        thisLB = m._vars[i].lb
        if thisLB == -0.0:
            thisLB = 0.0
        thisUB = m._vars[i].ub
        if math.isinf(thisUB):
            thisUB = 1000
        if m.ModelSense == GRB.MINIMIZE:
            thisOBJ = m._vars[i].obj
        else:
            thisOBJ = -(m._vars[i].obj)
        if firstNode:
            firstNode = False
            nodeFeatureVectors = np.array([thisType, thisLB, thisUB, thisOBJ, 0.0, 0.0])
        else:
            nodeFeatureVectors = np.vstack(( nodeFeatureVectors, np.array([thisType, thisLB, thisUB, thisOBJ, 0.0, 0.0]) ))
    src = []
    dst = []
    indexByVar = {v: i for i, v in enumerate(m._vars)}
    firstEdge = True
    for i in range(len(m._cons)):
        conNodeID = numberOfVars + i
        thisRHS = m._cons[i].RHS
        thisSense = senseMap[m._cons[i].Sense]
        nodeFeatureVectors = np.vstack(( nodeFeatureVectors, np.array([0.0, 0.0, 0.0, 0.0, thisRHS, thisSense]) ))

    # Compile edge feature vectors
        expr = m.getRow(m._cons[i])
        for j in range(expr.size()):
            varNodeID = indexByVar[expr.getVar(j)]
            thisCoeff = expr.getCoeff(j)
            src.append(conNodeID)
            dst.append(varNodeID)
            if firstEdge:
                firstEdge = False
                edgeFeatureVectors = np.array([thisCoeff])
            else:
                edgeFeatureVectors = np.vstack((edgeFeatureVectors, np.array([thisCoeff])))
            src.append(varNodeID)
            dst.append(conNodeID)
            edgeFeatureVectors = np.vstack(( edgeFeatureVectors, np.array([thisCoeff]) ))

    # Add one self-loop edge for every variable - let the edge attribute be equal to 1.0 so that it send's its info to itself; this is to account for isolated variables
    for i in range(nnodes):
        src.append(i)
        dst.append(i)
        edgeFeatureVectors = np.vstack(( edgeFeatureVectors, np.array([1.0]) ))

    src = np.asarray(src)
    dst = np.asarray(dst)

    # Create the actual graph object using DGL
    g = dgl.DGLGraph()
    g.add_nodes(nnodes)
    g.ndata['h'] = torch.as_tensor(nodeFeatureVectors)
    g.add_edges(src,dst)
    g.edata['w'] = torch.as_tensor(edgeFeatureVectors)
    try:
        problemType = modelLabels[model]
    except:
        problemType = -1
    graph_labels = {'glabel': torch.tensor([problemType])}
    firstSlash = models[model].find("/")
    lastSlash = models[model].rfind("/")
    save_loc = './ILP_graphs/'+models[model][firstSlash+1:lastSlash+1]
    if not os.path.isdir(save_loc):
        os.makedirs(save_loc)
    save_graphs(save_loc+model+".bin", g, graph_labels)
