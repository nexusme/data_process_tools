import pandas as pd
import numpy as np
from sklearn.preprocessing import add_dummy_feature


def dummy_features(df, value):
    add_dummy_feature(df, value=value)
    return df


# if __name__ == '__main__':
#     df_1 = pd.DataFrame([[1, 2, 1, 0],
#                          [3, 4, 6, 1],
#                          [2, 9, 6, 5],
#                          [9, 3, 7, 4]],
#                         columns=list('ABCD'))
#     print(df_1)
#     d = dummy_features(df_1, 2.0)
#     print(d)
