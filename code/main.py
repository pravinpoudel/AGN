from ast import arg
from cgi import test
from distutils.command.config import config
# from importlib.resources import path
import os
from pickle import TRUE
from platform import node
from statistics import mode
import yaml

import argparse
from symbol import argument
import networkx as nx
# import node2vec
import pandas as pd
# make the loop shows progress
from tqdm import tqdm

import numpy as np
# pytorch related package
import torch
import torch.nn as nn # contain neural network packages
import torch.nn.functional as F

# boss is here !!
from torch_geometric.data import Data
#import training model class
# from models import GCNL, GAT 

DATASET_DIR = '/work/GNN/original_data'
_seperator = "\t"
# modelsOption = {
#     'GCNL' : GCNL,
#     'GAT' : GAT,
# }

def retriveData(dataDir):
    myDirName = os.path.join(DATASET_DIR, dataDir, "train.data") 
    # the schema is stored in the config file inside the directory
    # We don't need to explicitly call the close() method. It is done internally.
    with open(os.path.join(myDirName, "config.yml"), "r") as stream:
        try:
            conf_data = yaml.safe_load(stream)
            print("---------------------------------------configuration file opened successfully ------------------------------------------------")
        except yaml.YAMLError as exc:
            print(exc)
    
    # read tsv file and get pandas.DataFrame
    try: 
        edgeTSV = pd.read_csv(os.path.join(myDirName, 'edge.tsv'), sep=_seperator, header=0)
        featuresTSV = pd.read_csv(os.path.join(myDirName, 'feature.tsv'), sep=_seperator, header=0)
    except IOError as err:
        print(err)
    
    # now iterate over all the dataframe
    edgeIterator = edgeTSV.iterrows()
    featureIterator = featuresTSV.iterrows()

    edges = []
    features = []

    for index, row in edgeIterator: #it does not have index of the row
        edges.append(row.to_list()) # since the data that we get from each row is source, destination and weight index so just wanted to get values not entries
    

    for index, row in featureIterator: #it has row index but it doesnot matter because array index will be enough
        fd = row.to_list()[1:]
        if len(fd) < 100:
            t = 100 - len(fd)
            fd = np.pad(fd, padWidth = (0, t), mode="constant")
        features.append(fd) # slice from second element
    
    assert len(features) == featuresTSV.shape[0] #check if we appended all?

    # shit, shape() is in tensor
    featureLength = len(features[0])
    edgeLength = len(edges)
    conf_data["edge_count"] = edgeLength

    # we can not feed array into pytorch so transform these to Tensor !! 
    # we could also do that with transform params in data loader of torch_geometry data as toTensor() where toTensor can be class with __init__ doing transformation
    # https://www.youtube.com/watch?v=X_QOZEko5uE&ab_channel=PythonEngineer

    # transpose is smart way of getting [[sources], [destination], [weight]]
    edges = torch.Tensor(edges).long().t()
    edge_index = edges[[0,1]]
    edges_attr = edges[[2]].t().float() #right now our data has weight of 1 but just for decimanl weight as well so casted to float


    # check before writing into config file
    
    conf_data["feature_count"] = featureLength

    features = torch.Tensor(np.array(features, dtype=float))
    # number of feature is number of node
    conf_data["n_node"] = features.shape[0]

    # load train label and test lable
    try: 
        test_labels = pd.read_csv(os.path.join(myDirName, 'test.csv'), sep=_seperator, header=0)
        train_labels = pd.read_csv(os.path.join(myDirName, 'train.csv'), sep=_seperator, header=0)
    except IOError as err:
        print(err)
    
    labels = torch.zeros(features.shape[0]).long()
    ratio_VD = 0.15 #TODO: check if there is env file possible in python so that we can change this kind of value from a file
    # create empty tensor to store the value in prediction
    # we can store lable of vertex in an array because same data is divided into two part
    train_mask = np.zeros(features.shape[0],  dtype=bool)
    test_mask = np.zeros(features.shape[0],  dtype=bool)
    validation_mask = np.random(features.shape[0]) < ratio_VD
    _class = [];
    for index, row in test_labels.iterrows():
        node1, label1 = row.tolist() 
        labels[node1] = label1
        test_mask[node1] = 1
        _class.append(label1) if label1 not in _class else _class

    # Main Logic here is: https://stackoverflow.com/questions/2451386/what-does-the-caret-operator-do
    assert False == True^True
    # we need to make false in the validation mask which is test data so False = True * False
    validation_mask *= train_mask #now this is our Final validation mask
    # since we got validation mask from training data we need to update that mask as well 
    train_mask = train_mask ^ validation_mask

    for index, row in train_labels.iterrows():
        node2, label2 = row.tolist()
        labels[node2] = label2 
        train_mask[node2] = 1
        _class.append(label2) if label2 not in _class else _class
    
    conf_data["class_count"] = len(_class)
    print("number of labels - ", len(_class))


    return conf_data, edge_index, edges_attr, features, labels, train_mask, validation_mask, test_mask


def main(args):
    # print(args)
    # if this is running for first time in the dataset run the code otherwise use stored processed dataset from the multiple dataset files
    loadedFile = os.path.join(DATASET_DIR, args.dataset, "load.pt")
    if not os.path.exists(loadedFile):
        conf_data, edge_index, edges_attr, features, labels, train_mask, validation_mask, test_mask = retriveData(args.dataset)
        torch.save([conf_data, edge_index, edges_attr, features, labels, train_mask, validation_mask, test_mask], loadedFile)
    else:
        conf_data, edge_index, edges_attr, features, labels, train_mask, validation_mask, test_mask = torch.load(loadedFile)



if __name__ == "__main__":
    print("------------------------------------------------------")
    print ("main.py execuated as main")
    # parse the args
    parser = argparse.ArgumentParser(description=' retrive process option argument')
    parser.add_argument("--dataset", choices=["a", "b", "c", "d", "e", "f"], required=True)
    parser.add_argument("--M", choices=["GNNL", "GAT"])
    parser.add_argument("--epoch", type=int, default=100) #defualt number of pass is 100
    _args = parser.parse_args()
    main(_args)

else:
    print ("i am executed from imported")



