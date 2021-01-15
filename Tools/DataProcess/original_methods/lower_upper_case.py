import pandas as pd
import numpy as np


def lower_upper_case(df, col, type_r):
    if type_r == 'upper':
        df[col] = df[col].str.upper()
    elif type_r == 'lower':
        df[col] = df[col].str.lower()
    return df


# if __name__ == '__main__':
#     df_1 = pd.DataFrame([[1, 'B', np.nan, 0],
#                          [3, 'b', np.nan, 1],
#                          [2, np.nan, np.nan, 5],
#                          [8, 'c', np.nan, 4]],
#                         columns=list('ABCD'))
#     print(df_1)
#     d = lower_upper_case(df_1, 'B', 'upper')
#     print(d)
