import matplotlib.pyplot as plt
import numpy as np

# 产生50个-1到2的数据点
x = np.linspace(-1,2,50)
y = 2*x + 1
y2 = x**2
# 开始做一张图
plt.figure(num=3, figsize=(8,5))
# 描点
l1, = plt.plot(x, y, label='up')
l2, = plt.plot(x, y2, color='red',linewidth='3', linestyle='--', label='down')
# 设置图例，loc还可以取loc="upper left"
plt.legend(handles=[l1,l2,], labels=['aaa','bbb'],loc='best')
# 设置x的取值范围
plt.xlim((-1, 2))
# 设置x轴的名称
plt.xlabel('iamx')
# 设置y的小标及别名
plt.yticks([-1, 0, 1],['bad', 'normal', r'$good\alpha$'])
# 获取整个坐标轴
ax = plt.gca()
# 将右轴设置成透明
ax.spines['right'].set_color('none')
# 将下轴设置为x轴
ax.xaxis.set_ticks_position('bottom')
# 将左轴设置成有y轴
ax.yaxis.set_ticks_position('left')
# 下轴的位置在0位置
ax.spines['bottom'].set_position(('data', 0))
# 左轴的位置在0位置
ax.spines['left'].set_position(('data', 0))
# 设置坐标轴点的背景box
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(12)
    label.set_bbox(dict(facecolor='white',edgecolor='None', alpha=0.7))

# 做点标注
x0 = 1
y0 = 2*x0 + 1
plt.scatter(x0, y0)
plt.plot([x0,x0], [0,y0], 'k--', lw=2.5)
# 标注方法一
# 做点的标注，arrowprops参数很多可以调整
plt.annotate(r'$2x+1=%s$' % y0, xy=(x0,y0), xycoords='data', xytext=(+30,-30), textcoords='offset points', fontsize=16, arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))
# 标注方法二
plt.text(1.2, 2, r'$test$')
plt.show()