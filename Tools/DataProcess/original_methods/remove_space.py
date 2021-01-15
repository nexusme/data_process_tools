import pandas as pd
import numpy as np


def remove_space(df, col):
    newName = df[col].str.strip()
    df[col] = newName
    return df


# if __name__ == '__main__':
#     df_1 = pd.DataFrame([[1, '          B', np.nan, 0],
#                          [3, 'b', np.nan, 1],
#                          [2, np.nan, np.nan, 5],
#                          [8, 'c', np.nan, 4]],
#                         columns=list('ABCD'))
#     print(df_1)
#     d = remove_space(df_1, 'B')
#     print(d)
