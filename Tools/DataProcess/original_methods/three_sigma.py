import pandas as pd
import numpy as np


def three_sigma(df, col):
    ser = df[col]
    bool_id = ((ser.mean() - 3 * ser.std()) <= ser) & (ser <= (ser.mean() + 3 * ser.std()))
    bool_dict = dict(bool_id)

    true_list = [key for key, value in bool_dict.items() if value]
    false_list = [key for key, value in bool_dict.items() if not value]

    df_true = df.iloc[true_list[0]:true_list[-1] + 1]
    if len(false_list) != 0:
        df_false = df.iloc[false_list[0]:false_list[-1] + 1]
    else:
        df_false = pd.DataFrame()

    print(df_true)
    print(df_false)

    return df_true, df_false


if __name__ == '__main__':
    df_1 = pd.DataFrame([[1, 2, np.nan, 0],
                         [3, 4, np.nan, 1],
                         [2, np.nan, np.nan, 5],
                         [8, 3, np.nan, 4]],
                        columns=list('ABCD'))
    # print(df_1)
    d = three_sigma(df_1, 'A')
    # print(d)
