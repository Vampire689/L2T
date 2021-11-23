import pandas as pd
import numpy as np
import math
from scipy.linalg import expm,logm
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from collections import Counter

# matplotlib.rcParams['font.family'] = 'SimHei'

fig = plt.figure(figsize=(6.2, 3.8))
titles=['./results/cifar10_2_255_result.xlsx','./results/cifar10_8_255_result.xlsx',
        './results/mnist_0.1_result.xlsx','./results/mnist_0.3_result.xlsx']
datas=[]
head1=list()
for j in range(len(titles)):
    data = pd.read_excel(titles[j])
    head = data.columns[:8]
    length=len(data.values[:,0])
    while not isinstance(data.values[length-1,0],int):
        length-=1
    for k in range(length):
        datas.append(data.values[k,:8])

    head1 = list(head)
head1[1] = 'Nnenum'
head1[4] = 'BaBSR'
head1[5] = 'L2T'
# print(type(datas))
# print(datas[0])
datas = np.mat(datas)
y = list(range(1, datas.shape[0]+1))

ls = ['-.', ':', '--', '-', '-']
lw = [2.5, 4, 2.5, 2.5, 2.5]
color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']
for idx, i in enumerate([1,2,4,3,5]):
    tmp=np.squeeze(datas[:,i],axis=1)
    tmp.sort()
    tmp = tmp.tolist()
    tmp = tmp[0]
    tmp = [400 if d>=300 else d for d in tmp]
    # tmp = np.transpose(tmp)
    # for m in range(tmp.shape[0]):
    #     for n in range(tmp.shape[1]):
    #         tmp[m,n] = 0.5*tmp[m,n]
    # plt.semilogy(y, tmp, linestyle=ls[idx], label=head1[i], linewidth=lw[idx], basey=2)
    plt.plot(y, tmp, linestyle=ls[idx], label=head1[i], linewidth=lw[idx], color=color[idx])
plt.vlines(337,0,300,colors = "k",linestyles = "dashed")

plt.legend(loc=4,fontsize=9)
plt.ylabel('Time(s)')
plt.xlabel('Verifiable properties')
plt.ylim(0,300)   
plt.xlim(50,350)
plt.tight_layout()

fig.savefig('nonTarget.pdf', format='pdf')