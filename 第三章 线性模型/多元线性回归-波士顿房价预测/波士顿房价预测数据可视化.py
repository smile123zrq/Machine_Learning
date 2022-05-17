import tensorflow as tf
import matplotlib.pyplot as plt

boston_housing = tf.keras.datasets.boston_housing

# 获取测试数据集 test_split = 0 表示不需要测试的数据（原数据分为测试数据和训练数据）
(train_x, train_y), (_, _) = boston_housing.load_data(test_split=0)
# 要显示中文，就必须设置中文字体，此处为黑体
plt.rcParams['font.sans-serif'] = ['SimHei']

# 正常显示负号
plt.rcParams['axes.unicode_minus'] = False
# 设置13个散点图对应的标题
titles = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B-1000', 'LSTAT', 'MEDV']
# 画布大小
plt.figure(figsize=(10, 10))
# 该条目下共有13个条目，共循环13次
for i in range(13):
    plt.subplot(4, 4, (i + 1))
    # 绘制散点图，s为表中图例的大小，marker设置图例为点
    plt.scatter(train_x[:, i], train_y, s=10, c='k', marker='.')
    plt.xlabel(titles[i])
    plt.ylabel("Price($1000's)")
    plt.title(str(i + 1) + '.' + titles[i] + '-Price')
plt.tight_layout()
# 调整显示的图表大小
plt.rcParams['figure.figsize'] = (10.0, 10.0)
# 调整标题位置
plt.suptitle("各个属性与房价之间的关系", x=0.5, y=1.00, fontsize=10)
plt.show()
