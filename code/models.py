import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import sklearn.metrics as metrics
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TAGConv
# GCNConv is a class not functon so we have to create the instance of this class in init

#TODO: rather than putting seperate name for each convolution put all the convolution in an array and iterate that array like this
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial7/GNN_overview.html
# 12
#TODO: Use Sequential
# https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html

#TODO: Add reset for model weight and parameters

# Architecture:
# https://stats.stackexchange.com/questions/281601/if-your-regression-model-is-not-over-fitting-would-using-regularization-give-be

# global_mean_pool - return batch-wise graph-lvel-outputs by averaging node features



class GCNN(nn.Module): #dropout p default value is 0.5 anyway in Dropout function
    def __init__(self, numb_classfic, input_numbers = 100, hidden_layers = 64, drop_out_rate = 0.5, has_edge_feature = True ):
        super(GCNN, self).__init__()
        # feature extraction
        self.layer1Conv = GCNConv(input_numbers, hidden_layers)
        self.layer2Conv = GCNConv(hidden_layers, hidden_layers)
        self.layer3Linear = nn.Linear(hidden_layers, numb_classfic)
        # check if we are having edge weight or not
        self.drop = drop_out_rate
    
    # what is softmax activation function
    # https://www.youtube.com/watch?v=8ah-qhvaQqU

    # softmax function calculate probability for each class in which input belongs so number of softmax output should always match
    # number of class so mathematically you can see this as a probability distribution
    # softmax function convert real values into proabilities 
    # it is only used as output layer of neural network
    # you can consider higher proability as actual output

    def resetParameter(self):
        # return super().register_parameter(name, param)
        self.layer1Conv.reset_parameters()
        self.layer2Conv.reset_parameters()
        # self.layer1Conv.reset_parameters()

    def forward(self, data):
        g, edge_index = data.x, data.edge_index
        g = self.layer1Conv(g, edge_index)
        g = F.relu(g)
        # right now value is 0.5 and default is also 0.5 so i am not passing this value
        g = F.dropout(g, training=self.training)
        g = self.layer2Conv(g, edge_index)
        g = F.relu(g)
        g = F.dropout(g, training=self.training)
        # why do we do linear ? 
        # https://www.sharetechnote.com/html/Python_PyTorch_nn_Linear_01.html
        g = self.layer3Linear(g)
        # no softmax at the end because we are going to use cross entropy
        return F.log_softmax(g, dim=1)
        # nn.Softmax() creates a modile and return function whereas F's are pure function
        # return F.log_softmax(g) # F.log_softmax is numerically more stable https://discuss.pytorch.org/t/what-is-the-difference-between-log-softmax-and-softmax/11801/8
    

class GAT(nn.Module):
    def __init__(self, numb_classfic, input_numbers = 100, hidden_layers = 64, drop_out_rate = 0.5, has_edge_feature = True) :
        super(GAT, self).__init__()
        self.layer1Conv = GATConv(input_numbers, hidden_layers, heads = 8, dropout=drop_out_rate, concat=False)
        self.layer2Conv = GATConv(hidden_layers, hidden_layers, heads = 8, dropout=drop_out_rate, concat=False)
        self.linear = nn.Linear(hidden_layers,numb_classfic)
        self.drop = drop_out_rate

    def resetParameter(self):
        self.layer1Conv.reset_parameters()
        self.layer2Conv.reset_parameters()
        nn.init.normal_(self.linear.weight)
        nn.init.normal_(self.linear.bias)

    def forward(self, data):
        g, edge_index = data.x, data.edge_index
        g  = self.layer1Conv(g, edge_index)
        g  = F.relu(g)
        g  = F.dropout(g, p= self.drop, training = self.training)
        g  = self.layer2Conv(g, edge_index)
        g  = F.relu(g)
        g  = F.dropout(g, p= self.drop, training=self.training)
        g = self.linear(g)
        return F.log_softmax(g, dim=1)    
    

class SAGE(nn.Module):
    def __init__(self, numb_classfic, input_numbers = 100, hidden_layers = 64, drop_out_rate = 0.5, has_edge_feature = True):
        super(SAGE, self).__init__()
        self.layer1Conv = SAGEConv(input_numbers, hidden_layers) 
        self.layer2Conv = SAGEConv(hidden_layers, hidden_layers)
        self.layer3Conv = SAGEConv(hidden_layers, numb_classfic)
        self.linear = nn.Linear(hidden_layers, numb_classfic)
        self.drop_rate =   drop_out_rate
    
    def resetParameter(self):
        self.layer1Conv.reset_parameters()
        self.layer2Conv.reset_parameters()
        self.layer3Conv.reset_parameters()
        nn.init.normal_(self.linear.weight)
        nn.init.normal_(self.linear.bias)
    
    def forward(self, data):
        g, edge_index = data.x, data.edge_index
        g = self.layer1Conv(g, edge_index)
        g = F.relu(g)
        g = F.dropout(g, p= self.drop_rate, training=self.training)
        g = self.layer2Conv(g, edge_index)
        g = F.relu(g)
        g = F.dropout(g, p= self.drop_rate, training=self.training)
        g = self.layer3Conv(g, edge_index)
        # g = self.layer3Conv(g, edge_index)
        return F.log_softmax(g, dim=1)

class TAG(nn.Module):
    def __init__ (self, numb_classfic, input_numbers = 100, hidden_layers = 64, drop_out_rate = 0.5, k=3, has_edge_feature = True):
        super(TAG, self).__init__()
        self.layer1Conv = TAGConv(input_numbers, hidden_layers, K=k) 
        self.layer2Conv = TAGConv(hidden_layers, hidden_layers, K=k)
        self.layer3Conv = TAGConv(hidden_layers, numb_classfic, K=k)
        self.drop_rate = drop_out_rate

    def resetParameter(self):
        self.layer1Conv.reset_parameters() 
        self.layer2Conv.reset_parameters()
        self.layer3Conv.reset_parameters()

        # self.linear =  nn.Linear(hidden_layers, numb_classfic, K=k)
       
    def forward(self, data):
        g, edge_index = data.x, data.edge_index
        g = self.layer1Conv(g, edge_index)
        g = F.relu(g)
        g = F.dropout(g, p = self.drop_rate, training=self.training)
        g = self.layer2Conv(g, edge_index)
        g = F.relu(g)
        g = F.dropout(g, p = self.drop_rate, training=self.training)
        g = self.layer3Conv(g, edge_index)
        return F.log_softmax(g, dim=1)
    #  dont know about TAG nned to check documentation and examples