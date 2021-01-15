import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def outliers_proc(data, col_name, scale=3):
    """
    用于清洗异常值，默认用 box_plot（scale=3）进行清洗
    :param data: 接收 pandas 数据格式
    :param col_name: pandas 列名
    :param scale: 尺度
    :return:
    """
    data_m = data.copy()
    data_series = data_m[col_name]

    iqr = scale * (data_series.quantile(0.75) - data_series.quantile(0.25))
    val_low = data_series.quantile(0.25) - iqr
    val_up = data_series.quantile(0.75) + iqr
    rule_low = (data_series < val_low)
    rule_up = (data_series > val_up)
    rule, value = (rule_low, rule_up), (val_low, val_up)

    index = np.arange(data_series.shape[0])[rule[0] | rule[1]]
    # print("Delete number is: {}".format(len(index)))
    data_m = data_m.drop(index)
    data_m.reset_index(drop=True, inplace=True)
    # print('boundaries:', min(data_m[col_name]), 'to', max(data_m[col_name]))
    return data_m
