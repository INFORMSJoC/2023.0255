import dgl
from dgl.data.utils import load_graphs
import dgl.function as fn
import torch
import torch.nn as nn
import sys
import numpy as np
import csv
import time

# Load ILP graphs (in DGL format)
def load_dgl_graph(path):
    glist, label_dict = load_graphs(path)
    g = glist[0]
    problemType = label_dict['glabel'][0].item()
    return (g,problemType)

# Message passing
class NodeTimesEdge(nn.Module):
    def __init__(self, in_num_node_feats, out_num_node_feats):
        super(NodeTimesEdge, self).__init__()
        self.linear = nn.Linear(in_num_node_feats, out_num_node_feats)
    def forward(self, g, inputs):
        g.update_all(fn.u_mul_e('h', 'w', 'weighted_node_feat'), fn.mean(msg='weighted_node_feat', out='h'))
        h = g.ndata['h']
        h = torch.tanh(h)
        return self.linear(h)

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim_1, n_classes, dropout_rate=0.5):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.edgescaling1 = NodeTimesEdge(in_dim, hidden_dim_1)
        self.classify = nn.Linear(hidden_dim_1, n_classes)
    def forward(self, g):
        h = g.ndata['h']
        h = self.dropout(h)
        h = self.edgescaling1(g, h)
        g.ndata['h'] = h
        meanDenseRep = dgl.mean_nodes(g, 'h')
        pred = self.classify(meanDenseRep)
        return pred

if __name__ == "__main__":
    device = torch.device("cpu")
    labelMap = {'vrptw': 0, 'cwlp-ss': 1, 'pcp': 2, 'scheduling': 3, 'cvrp': 4, 'clp': 5, 'inr': 6, 'lotsizing': 7, 'coloring': 8, 'tup': 9, 'bppif': 10, 'maplabeling': 11, 'kps': 12, 'bpp2': 13, 'cuttingstock': 14, 'relaxedClique': 15, 'gap': 16, 'cpmp': 17, 'bpp': 18}
    ClassByID = {k:v for v,k in labelMap.items()}

    # print('\nTesting with the *TRAIN* set.\n')
    # time.sleep(2)
    # testpaths = [l.strip().split(",")[1] for l in open('train_set.csv').readlines()]
    # testclass = [l.strip().split(",")[2] for l in open('train_set.csv').readlines()]

    print('\nTesting with the *TEST* set.\n')
    time.sleep(2)
    testpaths = [l.strip().split(",")[1] for l in open('test_set.csv').readlines()]
    testclass = [l.strip().split(",")[2] for l in open('test_set.csv').readlines()]

    testset = [load_dgl_graph(entry) for entry in testpaths]

    # convert node and edge dtypes from float64 to float32
    for t in testset:
        g = t[0]
        g.ndata['h'] = g.ndata['h'].float()
        g.edata['w'] = g.edata['w'].float()

    # Define GCN hyperparameters and initialize the model
    num_node_features = 6
    num_hidden_1 = 256
    num_classes = 19
    model = Classifier(num_node_features, num_hidden_1, num_classes)

    # Load pre-trained model weights and begin testing
    model.load_state_dict(torch.load('trained_GCN.pt'))
    model.eval()
    print('\nInitiating testing procedure.')
    correctPredictions = 0.0
    for i in range(len(testset)):
        print('Test iteration: {}/{}'.format(i+1,len(testset)))
        testinstance = testset[i]
        test_bg = testinstance[0]
        probs_Y = torch.softmax(model(test_bg), 1)
        argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
        realClass = testclass[i]
        predClass = ClassByID[int(argmax_Y[0])]
        if predClass == realClass:
            correctPredictions += 1

    predictionAccuracy = (correctPredictions/len(testset))*100.0
    print('Prediction Accuracy: {:.4f}'.format(predictionAccuracy))
    print('Correct: {} / {}'.format(int(correctPredictions), len(testset)))
