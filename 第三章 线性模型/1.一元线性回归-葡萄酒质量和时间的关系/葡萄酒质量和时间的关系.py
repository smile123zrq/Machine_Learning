import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 载入数据, np.genfromtxt 从文本文件中加载数据，以 ‘，’ 为分割符,返回列表
data = np.genfromtxt('linear.csv', delimiter=',')
# print(data)

# 画图
'''
二维列表
[[         nan          nan]
 [ 77.          79.77515201]
 [ 21.          23.17727887]
 [ 22.          25.60926156]]
data[ a , b ] a的位置限制第几行，b的位置限制第几列, “ : ”表示全部数据
data[:,0]表示第1列所有数据
data[1,:]表示第2行所有数据
data[:, 1:]表示从第2列开始所有数据
'''

'''
# 散点图，传入两个列表，作为x与y的数据
plt.scatter(data[1:, 0], data[1:, 1], c='k', marker='.')   # 散点图，以为第一行不是数据，所以要去除第一行，从第二行开始
plt.title('Age VS Quality')
plt.xlabel('Age')
plt.ylabel('Quality')
plt.show()
'''

# 数据拆分，随机划分为训练集与测试集，30%为测试集
x_train, x_test, y_train, y_test = train_test_split(data[1:, 0], data[1:, 1], test_size=0.3)
# 这里获得数据是一个一维列表，x_train [ 15.  65.  61.  96.  54.  75.  20.] ,以‘ (空格)’ 分隔，'.'后面是小数
# print('x_train', x_train)
# print('y_train', y_train)

# 1D -> 2D，给数据增加一个维度，主要是训练模型时，函数要求传入2维数据
x_train = x_train[:, np.newaxis]
x_test = x_test[:, np.newaxis]

# 使用函数训练模型
model = LinearRegression()
model.fit(x_train, y_train)     # 在这里就是使用函数训练模型了
# LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

# 画图，训练集的散点图与线性回归折线图
plt.scatter(x_train, y_train, c='k', marker='.')
# model.predict(x_train)这里是使用训练出来的模型预测值，这里画的折线图使用训练所得的模型画出来的，而不是使用原数据
plt.plot(x_train, model.predict(x_train), c='r')    # model.predict(x_train)
plt.title('Age VS Quality(Training set Use function)')
plt.xlabel('Age')
plt.ylabel('Quality')
plt.show()

# 画图，训练集的散点图与线性回归折线图
plt.scatter(x_test, y_test, c='k', marker='.')
plt.plot(x_test, model.predict(x_test), c='r')
plt.title('Age VS Quality(Test set Use function)')
plt.xlabel('Age')
plt.ylabel('Quality')
plt.show()

# 使用原理公式来训练模型
# 划分数据
x_train2, x_test2, y_train2, y_test2 = train_test_split(data[1:, 0], data[1:, 1], test_size=0.3)


# 使用算法原理构造模型， f(Wi) = WXi + b  要得出这里的 w 与 b
x_train2_average = sum(x_train2) / len(x_train2)

temp1 = 0
for i in range(len(x_train2)):
    temp1 = temp1 + y_train2[i] * (x_train2[i] - x_train2_average)
temp2 = 0
for i in range(len(x_train2)):
    temp2 = temp2 + x_train2[i] * x_train2[i]
temp3 = 0
for i in range(len(x_train2)):
    temp3 = temp3 + x_train2[i]


w = temp1 / (temp2 - temp3*temp3 / len(x_train2))

temp4 = 0
for i in range(len(x_train2)):
    temp4 = temp4 + (y_train2[i] - w * x_train2[i])

b = temp4 / len(x_train2)

x = np.arange(0, 100)
y = w*x + b

# 画图，训练集的散点图与线性回归折线图
plt.scatter(x_train2, y_train2, c='k', marker='.')
plt.plot(x, y, c='r')
plt.title('Age VS Quality(Training set Use algorithm)')
plt.xlabel('Age')
plt.ylabel('Quality')
plt.show()

# 画图，测试集的散点图与线性回归折线图
plt.scatter(x_test2, y_test2, c='k', marker='.')
plt.plot(x, y, c='r')
plt.title('Age VS Quality(Test set Use algorithm)')
plt.xlabel('Age')
plt.ylabel('Quality')
plt.show()

