import pandas as pd
import numpy as np
from scipy import stats


class DataAnalysis(object):

    @staticmethod
    def trend(df, col_x, col_y):
        """
        计算时间序列之间的趋势
        :param df:
        :param col_x:
        :param col_y:
        :return:
        """
        x, y = np.mat(df[col_x]), np.mat(df[col_y])
        t = ((len(df[col_x]) * x @ y.T) - np.sum(x) * np.sum(y)) / (len(df[col_x]) * x @ x.T - np.power(np.sum(x), 2))
        return np.float(t)

    @staticmethod
    def ks_value(df, cols):
        """
        检查连续数据是否满足正态分布
        :param df:
        :param cols:
        :return:
        """
        kept_columns = []
        for name in cols:
            u = df[name].mean()  # 计算均值
            std = df[name].std()  # 计算标准差
            # print(list(stats.kstest(df[name], 'norm', (u, std)))[1])
            if list(stats.kstest(df[name], 'norm', (u, std)))[1] > 0.05:
                kept_columns.append(name)
        # print(kept_columns)
        return kept_columns

# if __name__ == '__main__':
#     df_1 = pd.DataFrame([[1, 126677],
#                          [2, 149045],
#                          [3, 195227], ],
#                         columns=['m', 'val'])
#     DataAnalysis.trend(df_1, 'm', 'val')
