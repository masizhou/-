import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def minmax_demo():
    """
    归一化演示
    """
    data = pd.read_csv("./data/dating.txt")
    #     print(data)

    # 1.实例化
    transfer = MinMaxScaler(feature_range=(0, 1))  # 归一化到3-5之间

    # 2.进行转换，调用fit_transform
    ret_data = transfer.fit_transform(data[["milage", "milage", "Consumtime"]])
    print("归一化之后的数据：\n", ret_data)


minmax_demo()


def standard_demo():
    """
    标准化演示
    """
    data = pd.read_csv("./data/dating.txt")
    #     print(data)

    # 1.实例化
    transfer = StandardScaler()  # 默认均值为0，标准差为1

    # 2.进行转换，调用fit_transform
    ret_data = transfer.fit_transform(data[["milage", "milage", "Consumtime"]])
    print("标准划之后的数据：\n", ret_data)
    print("每一列特征的平均值：\n", transfer.mean_)
    print("每一列特征的方差：\n", transfer.var_)

standard_demo()