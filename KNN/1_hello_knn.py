from sklearn.neighbors import KNeighborsClassifier

# 1.构造数据
x = [[1], [2], [10], [20]]
y = [0, 0, 1, 1]

# 2.训练模型
# 2.1实例化一个估计器对象
estimator = KNeighborsClassifier(n_neighbors=1)

# 2.2调用fit方法，进行训练
estimator.fit(x, y)

# 3.数据预测
ret = estimator.predict([[0]])
print(ret)

ret = estimator.predict([[100]])
print(ret)