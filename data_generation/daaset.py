from platform import node
import numpy as np
import pandas as pd
import yaml
import os

# train_id = 
# label = 

_seperator = '\t'
DATASET_DIR = '/work/GNN/cora'
citation = pd.read_csv(DATASET_DIR, "cora.edges", sep= "\t", header=None, names=["target", "source"])
node = pd.read_csv(DATASET_DIR, "cora.node_labels", sep= "\t", header=None, names=["node_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"] )

print(citation)  
# class Dataset:
#     def __init__(self, directory):
#         self.directory = directory
#         self.data = self.getDataset()
#         self.dataset_edges = None
#         self.edges = self.get_edge_data()
#     # def getDataset():
#     #     return {
#     #         "edge_file":
#     #         "feature_file":
#     #         "test_file":
#     #         "train_file":

#     #     }

#     def get_edge_data(self):
         
#         # myDataType = {
#         #     "src_indx": int,
#         #     "dst_indx": int,
#         #     "edge_weig": float
#         # }
#         # # check if it is not retrived before
#         # if self.edge_data is None:
#         #     # set self.dataset_edge 
#         #     # make tsv file <dataframe>
#         #     pd.read_csv(path, )
#         # return self.dataset_edges
