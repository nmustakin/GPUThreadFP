# GPUThreadFP

## Learn indirect memory access patterns of GPU Applications

### Files: 
* preprocessing.py - processes the raw trace texts and generates csv file as output
- usage: `python3 preprocessing.py traces/ld_bfs.txt`  *raw traces not provided due to large size

* visualize.py - visualize data accesses color coded for each thread block 
- usage: `python3 visualize.py processed_data/data_ld_bfs.csv`

* cluster.py - cluster to detect outliers and visualize thread blocks making indirect access 
- usage: `python3 cluster.py processed_data/data_ld_bfs.csv`

* deltaLSTM.py - train LSTM/SimpleRNN model and visualize the performance 
- usage: `python3 deltaLSTM.py [Filename] [mode=LSTM/RNN] [blocks] [lookback]` 
