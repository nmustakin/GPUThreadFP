import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import numpy as np
import sys
import matplotlib.cm as cm
from matplotlib.font_manager import FontProperties



if len(sys.argv) < 2:
  print('Format: python3 visualize.py [Filename]')
  sys.exit(0)
inputFile = str(sys.argv[1])
df = pd.read_csv(inputFile)

#print(df[['TB', 'tid', 'addr']])
#print(df)

array = df[['TB', 'tid', 'addr']].values


x = [[]]
y = [[]]
z = []
for data in array: 
  address = int(data[2])
  if int(data[0]) not in z:
    z.append(int(data[0]))
    x.append([])
    y.append([])

  print(int(address/16384)) 
  y[z.index(int(data[0]))].append(int(address/16384)) #32768
  x[z.index(int(data[0]))].append(address%16384) #32768
    
colors = cm.hsv(np.linspace(0, 1, 128))
np.random.shuffle(colors)
plots = []
#for i, c in zip(z, colors): 
#  plots.append(plt.plot(x[i], y[i], color=c, linestyle='None', markersize = 5.0, marker='.', label='TB '+str(i)))
  
  #plt.scatter(x[i], y[i], color=c)

for x_i, y_i, c in zip(x, y, colors):
  plt.plot(x_i, y_i, color=c, linestyle='None', markersize = 6.0, marker='.')

#plt.legend(plots, bbox_to_anchor=(1, 1), loc='upper left', fontsize='xx-small')
plt.show()