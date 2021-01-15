from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np


def poly_features(df, poly_num, inter_flag):
    poly = PolynomialFeatures(degree=poly_num, interaction_only=inter_flag)
    df = poly.fit_transform(df)
    return df


# if __name__ == '__main__':
#     df_1 = pd.DataFrame([[1, 2, 1, 0],
#                          [3, 4, 6, 1],
#                          [2, 9, 6, 5],
#                          [9, 3, 7, 4]],
#                         columns=list('ABCD'))
#     print(df_1)
#     d = poly_features(df_1, 2, True)
#     print(d)
