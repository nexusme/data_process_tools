import pandas as pd
import numpy as np


def nan_encoding(df, col):
    df[col] = df[col].apply(lambda x: 1 if np.isnan(x) == 1 else 0)
    return df
    # print(df[col])


# if __name__ == '__main__':
#     df_1 = pd.DataFrame([[np.nan, 2, np.nan, 0],
#                          [3, 4, np.nan, 1],
#                          [2, np.nan, np.nan, 5],
#                          [np.nan, 3, np.nan, 4]],
#                         columns=list('ABCD'))
#     d = nan_encoding(df_1, 'A')
