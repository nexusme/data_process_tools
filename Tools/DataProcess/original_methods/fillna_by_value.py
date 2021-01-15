import pandas as pd
import numpy as np


def fillna_by_value(df, columns, num):
    """
    缺失值填充： 使用指定数字填充
    :param df:
    :param columns:
    :param num: 指定数字
    :return:
    """
    df[columns] = df[columns].fillna(num)
    return df

# if __name__ == '__main__':
#     df_1 = pd.DataFrame([[np.nan, 2, np.nan, 0],
#                        [3, 4, np.nan, 1],
#                        [2, np.nan, np.nan, 5],
#                        [np.nan, 3, np.nan, 4]],
#                       columns=list('ABCD'))
#     print(df_1)
#     d = fillna_by_mean(df_1, 'A')
#     print(d)
