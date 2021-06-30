import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Compare plot_nsight.py dfs')
parser.add_argument('--batch', required=True )
parser.add_argument('--ifile', required=True )
parser.add_argument('--max_width', required=True )
parser.add_argument('--start_index', required=True )
parser.add_argument('--end_index', required=True )
parser.add_argument('--scale', required=True )
args = parser.parse_args()
args.batch = int(args.batch)
max_width = int(args.max_width)
start_idx = int(args.start_index)
end_index = int(args.end_index)
scale=int(args.scale)

df = pd.read_csv('{}'.format(args.ifile))
tmp_df = df[(df['LayerName']=='conv0') & (df['Phase']=='fprop')][['LayerName','GPUStartTime(ms)','GPUEndTime(ms)']]
l = tmp_df.to_records()
k = []
for i in range(start_idx,min(len(l)-1, end_index)):
    k.append(l[i+1][3] - l[i][2])
view = [] # 2d array like graph paper
k = [int(i/scale) for i in k]
minv = min(k)
height = int(max(k) + 5 - minv)
for h in range(0,height):
    view.append([])
for h in range(0, height):
    for w in range (0, len(k)):
        view[h].append(0)
for i in range(0, len(k)):
    view[int(k[i]-minv)][i] = 1
for h in range(height-1,-1,-1):
    string_s = ""
    for w in range (0, min(len(k),max_width)):
        if view[h][w] == 0:
            string_s += "|  "
        else:
            string_s += "| *"
    print(string_s)
string_s=""    
for w in range (start_idx, start_idx+min(len(k),max_width)):
    tmp = "|{:2}".format(str(w%100)).center(3)
    string_s+=str(tmp)
print(string_s)
