from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
import pandas as pd
import numpy as np


def rfecv(df, columns, target_col):
    X = df[columns]
    y = df[target_col]
    estimator = SVR(kernel="linear")
    selector = RFECV(estimator, step=1, cv=len(columns))
    selector = selector.fit(X, y)
    data = selector.transform(X)
    # get kept columns
    true_list = list(selector.get_support())
    index = [i for i in range(len(true_list)) if true_list[i] == True]
    saved_columns = [columns[i] for i in index]
    # save into dataframe
    result = pd.DataFrame(data, columns=saved_columns)
    result[target_col] = y
    return result

#
# if __name__ == '__main__':
#     df_1 = pd.DataFrame([['dog', 2, 1, 0, 1],
#                          ['dog', 4, 6, 1, 0],
#                          ['wolf', 9, 6, 5, 0],
#                          ['rabbit', 3, 7, 4, 1],
#                          ['dog', 4, 6, 1, 0],
#                          ['dog', 4, 6, 1, 0],
#                          ['wolf', 9, 6, 5, 0],
#                          ['rabbit', 3, 7, 4, 1],
#                          ['pig', 3, 9, 4, 1]
#                          ],
#                         columns=list('ABCDy'))
#     rfecv(df_1, ['D', 'B', 'C'], 'y')
