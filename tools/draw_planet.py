import matplotlib.pyplot as plt  #导入matplotlib库
import numpy as np  #导入numpy库
import mpl_toolkits.axisartist as axisartist
from matplotlib import rcParams
config = {
    "font.family":'serif',
    "font.size": 20,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)

#创建画布
fig = plt.figure(figsize=(7.5, 4))
#使用axisartist.Subplot方法创建一个绘图区对象ax
ax = axisartist.Subplot(fig, 111)  
#将绘图区对象添加到画布中
fig.add_axes(ax)

#通过set_visible方法设置绘图区所有坐标轴隐藏
ax.axis[:].set_visible(False)
#ax.new_floating_axis代表添加新的坐标轴
ax.axis["x"] = ax.new_floating_axis(0,0)
#给x坐标轴加上箭头
ax.axis["x"].set_axisline_style("-|>", size = 1.0)
#添加y坐标轴，且加上箭头
ax.axis["y"] = ax.new_floating_axis(1,0)
ax.axis["y"].set_axisline_style("-|>", size = 1.0)
#设置x、y轴上刻度显示方向
ax.axis["x"].set_axis_direction("top")
ax.axis["y"].set_axis_direction("right")

#生成x步长为0.1的列表数据
x = np.arange(-7,7,0.1)
#生成relu形式的y数据
y = [(i>0)*i for i in x]
y1 = [(x[-1]+i)*0.5 for i in x]
#设置x、y坐标轴的范围
plt.xlim(-9,9)
plt.ylim(-1, 9)
#绘制图形
plt.plot(x[:len(x)//2],y[:len(x)//2], c='black', lw=4)
plt.plot(x[len(x)//2:],y[len(x)//2:], c='black', lw=4)
plt.plot(x,y1, c='black', ls='--', lw=3)
plt.scatter(x[0],0, c='black', marker = '*', s=160)
plt.scatter(x[-1],0, c='black', marker = '*', s=160)
plt.fill_between(x,y,y1, facecolor='lightgreen',interpolate=True)
plt.text(x[0] - 1.3, -1.1, '$\mathbf{l}^{[i]}_{j}$', fontsize = 20)
plt.text(6.5 - 1, -1.1, '$\mathbf{u}^{[i]}_{j}$', fontsize=20)
plt.text(7.5 + 0.7, -1.1, '$\mathbf{\hat{x}}^{[i]}_{j}$', fontsize=20)
plt.text(-1.5, 9.5, '$\mathbf{x}^{[i]}_{j}$', fontsize=20)
plt.xticks([])
plt.yticks([])

# fig.savefig('planet.jpg', format='jpg')
fig.savefig('planet.pdf', format='pdf')