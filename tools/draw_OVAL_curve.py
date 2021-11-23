import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import matplotlib
# matplotlib.rcParams['font.family'] = 'SimHei'

fig = plt.figure(figsize=(6, 4))

datas = pd.read_excel('results/base_result.xlsx')
head = list(datas.columns[:18])
head[15] = "L2T"
data = datas.values[:100, :18]
y = list(range(1, int(len(data[:,0])+1)))

ls = ['-.', ':', '--', '-', '-']
lw = [2.5, 4, 2.5, 2.5, 2.5]
color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']

for idx, i in enumerate([9, 12, 6  , 17, 15]):
    tmp = list(data[:,i])
    tmp.sort()
    tmp = [4000 if d>=3600 else d for d in tmp]
    plt.semilogy(y, tmp, linestyle=ls[idx], label=head[i], linewidth=lw[idx], color=color[idx])
    # plt.plot(y, tmp, label=head[i])
plt.vlines(100,0,3600,colors = "k",linestyles = "dashed")
plt.legend(loc=4,fontsize=9)
plt.ylabel('Time(s)')
plt.xlabel('Complete percentage(%)')
plt.ylim(0,3600)    
plt.title("Base model")
# plt.show()
fig.savefig('baseResult.pdf', format='pdf')