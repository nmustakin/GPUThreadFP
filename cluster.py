import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.neighbors import LocalOutlierFactor

if len(sys.argv) < 2:
  print('Format: python3 visualize.py [Filename]')
  sys.exit(0)
inputFile = str(sys.argv[1])
df = pd.read_csv(inputFile)

addresses = df[['addr']].values

clf = LocalOutlierFactor(n_neighbors=26)
pred = clf.fit_predict(addresses)

x = [[], []]
y = [[], []]
for i in range(0, len(addresses)): 
  if pred[i] == 1: 
    x[0].append(int(addresses[i]%32768))
    y[0].append(int(addresses[i]/32768))
  else: 
    x[1].append(int(addresses[i]%32768))
    y[1].append(int(addresses[i]/32768))

plt.plot(x[0], y[0], label='contiguous', linestyle='None', marker='.', markersize = 6.0)
plt.plot(x[1], y[1], label='outlier', linestyle='None', marker='.', markersize = 8.0)

plt.legend()
plt.show()
