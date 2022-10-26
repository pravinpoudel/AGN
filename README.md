# AGN

1. No Softmax but Cross Entropy <br />
2. No Adam but SGD <br />
3. train, validation and test data or node in graph is stored with mask array for each holding True/False as boolean Value <br />
4. use load.pt as cache file to store the extracted data needed to make graph so that when data retriver is called again, it won't need to be called rather data can be loaded from
load.pt file

### Next thing to find out
How to send homogeneous graph dictionary by pytorch geometry in the training
