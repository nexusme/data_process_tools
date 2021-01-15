import pandas as pd
import numpy as np


def drop_columns(df, columns):
    """
    删除指定列
    :param df:
    :param columns:
    :return:

    """

    df = df.drop(columns, axis=1)
    return df


# if __name__ == '__main__':
#     df_1 = pd.DataFrame([[np.nan, 2, np.nan, 0],
#                          [3, 4, np.nan, 1],
#                          [2, np.nan, np.nan, 5],
#                          [np.nan, 3, np.nan, 4]],
#                         columns=list('ABCD'))
#     print(df_1)
#     d = drop_columns(df_1, 'A')
#     print(d)
