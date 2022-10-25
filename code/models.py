import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import sklearn.metrics as metrics
from torch_geometric.nn import GCNConv, GATConv
# GCNConv is a class not functon so we have to create the instance of this class in init

class GCNN(nn.nodule): #dropout p default value is 0.5 anyway in Dropout function
    def __init__(self, numb_classfic, input_numbers = 100, hidden_layers = 64, drop_out_rate = 0.5, has_edge_feature = True ):
        super(GCNN, self).__init__()
        # feature extraction
        self.layer1Conv = GCNConv(input_numbers, hidden_layers)
        self.layer2Conv = GCNConv(hidden_layers, hidden_layers)
        self.layer3Linear = nn.Linear(hidden_layers, numb_classfic)
        # check if we are having edge weight or not
        self.drop = drop_out_rate
    
    def forward(self, data):
        epochs = 100
    # feature extraction 
        for i in range(epochs):
            result = self.layer1Conv()
            result = F.relu(result)
            result = self.conv2(result, )
            result = #do relu()
    result= #do relu
    result = #do dropouut
         
    #classification 
    x= #linearlize the output to classify

    return #logvalue     
