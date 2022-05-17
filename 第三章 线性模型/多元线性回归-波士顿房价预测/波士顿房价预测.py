import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
import seaborn as sns

# 载入波士顿房屋的数据集
data = np.genfromtxt('housing.csv', delimiter=',')

# x为数据集,第一行为标签(不要)，最后一列为target(不要)
x = data[1:, 0: -1]
# y为target
y = data[1:, -1]

# 数据标准化
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()  # 将特征数据的分布调整成标准正太分布，也叫高斯分布，也就是使得数据的均值维0，方差为1.
x = ss.fit_transform(x)

# print(x[:5])

# 随机划分为训练集与测试集
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# 创建模型
model = LassoCV()
model.fit(x_train, y_train)

# lasso系数
print(model.alpha_)
# 相关系数
print(model.coef_)

print(model.score(x_test, y_test))

