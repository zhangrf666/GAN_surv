import pandas as pd
filename='/home/zrf2022/GANs/Merged_ElasticNetExprs.txt'

data=pd.read_table(filename)
print(data.isnull().values.any())
print(data.shape)