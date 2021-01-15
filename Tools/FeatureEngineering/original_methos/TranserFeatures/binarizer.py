from sklearn.preprocessing import Binarizer
import pandas as pd
import numpy as np


def binarizer(df):
    X = df.values
    transformer = Binarizer().fit(X)  # fit does nothing.
    matrix = transformer.transform(X)
    return matrix


# if __name__ == '__main__':
#     df_1 = pd.DataFrame([[1, 2, 1, 0],
#                          [5, 4, 6, 1],
#                          [7, 9, 6, 5],
#                          [3, 3, 7, 4]],
#                         columns=list('ABCD'))
#     print(df_1)
#     d = binarizer(df_1)
#     print(d)
