import pandas as pd
import numpy as np

type_dict = {0: 'keep letter', 1: 'keep num', 2: 'keep letter and num'}


def remove_str_num(df, col, type_r):
    if type_r == 0:
        df[col] = df[col].apply(lambda x: ''.join(filter(str.isalpha, str(x))))

    elif type_r == 1:
        df[col] = df[col].apply(lambda x: ''.join(filter(str.isdigit, str(x))))

    elif type_r == 2:
        df[col] = df[col].apply(lambda x: ''.join(filter(str.isalnum, str(x))))
    return df


# if __name__ == '__main__':
#     df_1 = pd.DataFrame([[1, 'B01', np.nan, 0],
#                          [3, 'b200', np.nan, 1],
#                          [2, np.nan, np.nan, 5],
#                          [8, '222c', np.nan, 4]],
#                         columns=list('ABCD'))
#     print(df_1)
#     remove_str_num(df_1, 'B', 2)
#     # print(d)
