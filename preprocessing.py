# !/bin/python
from __future__ import print_function
#import numpy as np
import os
# import statistics
import sys
import pandas as pd
import re


target_kernel = 1
print("Here")

if len(sys.argv) < 2:
  print('Format: python3 extract_ld_info.py [Filename] [Target Kernel ID]')
  sys.exit(0)
ld_file = str(sys.argv[1])
if len(sys.argv) > 2:
  target_kernel = int(sys.argv[2])

#if not ld_file.startswith('ld_'):
#  sys.exit(0)

# Extract info
df = pd.DataFrame()
new_df = pd.DataFrame()

corr_data = []

archive = {}

count = 0
block_size = 2048
start_extract = False
with open(ld_file) as f:
    for line in f.readlines():
        if line[0] == 'K':
            elem = line.split(':')
            if elem[0][1:] == str(target_kernel):
                    start_extract = True
                    if elem[1] is not None:
                        block_size = int(elem[1].strip())
            #        print('Kernel {} found with block size {}'.format(target_kernel, block_size))
                    continue
            elif elem[0][1:] == str(target_kernel + 1):
                break
        if start_extract == False:
            continue

        # Extract data from line
        results = re.search('@PC=(.*); TB (.*), tid (.*): (.*) (.*) \(Idx (.*)\) => (.*)', line)
        data = {
                'pc': int(results.group(1), 16),
                'TB': int(results.group(2)),
                'tid': int(results.group(3)),
                'addr': int(results.group(5), 16),
                'addr_offset': int(results.group(6)),
            }


        # DataFrame
        if results.group(4)=='LD':
            df = df.append(data, ignore_index=True)
            #print(data)

df.to_csv('data_ld_vecAdd.csv')
print("Done")