import pandas as pd
import numpy as np


def rename_column(df_name, col, new_col):
    df_m = df_name
    df_m.rename(columns={col: new_col}, inplace=True)
    return df_m


# if __name__ == '__main__':
#     df_1 = pd.DataFrame([[1, 2, np.nan, 0],
#                          [3, 4, np.nan, 1],
#                          [2, np.nan, np.nan, 5],
#                          [8, 3, np.nan, 4]],
#                         columns=list('ABCD'))
#     print(df_1)
#     d = rename_column(df_1, 'A', 'a')
#     print(d)
