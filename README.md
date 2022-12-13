# AGN

use load.pt as cache file to store the extracted data needed to make graph so that when data retriver is called again, it won't need to be called rather data can be loaded from
load.pt file

#Regarding the original_file that i was sent as zip file, please unzip it and add it in main directory with same name.

This is my directory structure.

![image](https://user-images.githubusercontent.com/11494733/207249568-11aa0295-ba4f-4d03-b4dc-75c0bc599894.png)

and regarding that so many csv file, they are file that are used to store the training accuracy per epoch.
They are not named automatically because they are just for logging the data.

You can ignore it !!

Running Instruction: (while being on main folder)

```python
python code/main.py --dataset a --M GCN
```

dataset c is very very big so might take more than hours.

```python
--dataset attribute can be a, b or c 
--M attribute takes GCN, GAT, SAGE, TAG as value
```

### Note
```
requirements.txt file is inside the code/ folder
```
