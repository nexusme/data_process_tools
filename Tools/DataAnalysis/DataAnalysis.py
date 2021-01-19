import pandas as pd
import numpy as np


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


if __name__ == '__main__':
    df_1 = pd.DataFrame([[1, 126677],
                         [2, 149045],
                         [3, 195227], ],
                        columns=['m', 'val'])
    DataAnalysis.trend(df_1, 'm', 'val')
