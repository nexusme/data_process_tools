import pandas as pd
import numpy as np


def sort_values(df, columns):
    """
    数据排序
    :param df:
    :param columns:
    :return:
    """
    df = df.sort_values(by=[columns])
    return df


# if __name__ == '__main__':
#     df_1 = pd.DataFrame([[1, 2, np.nan, 0],
#                          [3, 4, np.nan, 1],
#                          [2, np.nan, np.nan, 5],
#                          [8, 3, np.nan, 4]],
#                         columns=list('ABCD'))
#     print(df_1)
#     d = sort_values(df_1, 'A')
#     print(d)
