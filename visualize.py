import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

df = pd.read_csv("processed_data/data_ld_vecAdd.csv")

#print(df[['TB', 'tid', 'addr']])
#print(df)

array = df[['TB', 'tid', 'addr']].values

x = [] 
y = []
### Each address occurs only once

# addresses = {}
# for data in array:
#     key = int(data[2])
#     if key in addresses:
#         addresses[key]+= 1
#     else: 
#         addresses[key] = 1 
# print(addresses)


### Memory addresses seem to be contiguous, there are no clusters 
# for data in array: 
#     address = int(data[2])
#     y.append(int(address/16192))
#     x.append(address%16192)
    
# plt.plot(x, y, 'r')

# plt.legend()
# plt.show()