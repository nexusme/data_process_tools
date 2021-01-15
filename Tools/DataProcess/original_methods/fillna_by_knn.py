import pandas as pd
import numpy as np
from fancyimpute import KNN


def fillna_by_knn(df):
    """
    KNN填充
    :param df:
    :return:
    """
    data = pd.DataFrame(KNN(k=6).fit_transform(df))
    data.columns = df.columns
    return data


# if __name__ == '__main__':
#     df_1 = pd.DataFrame([[np.nan, 2, np.nan, 0],
#                          [3, 4, np.nan, 1],
#                          [2, np.nan, np.nan, 5],
#                          [np.nan, 3, np.nan, 4]],
#                         columns=list('ABCD'))
#     d = fillna_by_knn(df_1)
#     print(d)
