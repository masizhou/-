import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.model_selection import train_test_split

# 显示中文
font = {'family':'SimHei', 'weight':'bold', 'size':'16'}
plt.rc('font', **font)               # 步骤一（设置字体的更多属性）
plt.rc('axes', unicode_minus=False)  # 步骤二（解决坐标轴负数的负号显示问题）

# 1.数据集获取
iris = load_iris()  # 小数据集，本地
# print(iris)
# news = fetch_20newsgroups()  # 大数据集，网上下载
# print(news)

# 2.数据集特征描述
print("数据集特征值是：\n", iris.data)
print("数据集目标值是：\n", iris["target"])
print("数据集的特征值名字：\n", iris.feature_names)
print("数据集的目标值名字：\n", iris.target_names)
print("数据集的描述：\n", iris.DESCR)

# 3.把数据转换成dataframe的格式
iris_d = pd.DataFrame(iris['data'], columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
iris_d['Species'] = iris.target  # 添加一列
print(iris_d)

# 4.画图
def plot_iris(iris, col1, col2):
    sns.lmplot(x = col1, y = col2, data = iris, hue = "Species", fit_reg = False)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('鸢尾花种类分布图')
    plt.show()

# plot_iris(iris_d, 'Petal_Width', 'Sepal_Length')

# 5.数据集划分
x_train,x_test,y_train,y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
# print("训练集的特征值是：\n", x_train)
# print("训练集的目标值是：\n", y_train)
# print("测试集的特征值是：\n", x_test)
# print("测试集的目标值是：\n", y_test)

print("训练集的特征值的形状是：\n", x_train.shape)
print("测试集的特征值的形状是：\n", x_test.shape)