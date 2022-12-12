# AGN

use load.pt as cache file to store the extracted data needed to make graph so that when data retriver is called again, it won't need to be called rather data can be loaded from
load.pt file

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
