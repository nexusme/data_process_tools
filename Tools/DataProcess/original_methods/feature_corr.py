import pandas as pd
import numpy as np


def feature_corr(df):
    """
    特征相关性
    :param df:
    """
    df.corr(method='pearson')
    print(df.corr(method='pearson'))
    return df.corr(method='pearson')


# if __name__ == '__main__':
#     df_1 = pd.DataFrame([[np.nan, 2, np.nan, 0],
#                          [3, 4, np.nan, 1],
#                          [2, np.nan, np.nan, 5],
#                          [np.nan, 3, np.nan, 4]],
#                         columns=list('ABCD'))
#     d = feature_corr(df_1)
#     print(d)
