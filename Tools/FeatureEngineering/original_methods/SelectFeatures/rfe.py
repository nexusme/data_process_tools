import pandas as pd
import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR


def rfe(df, columns, target_col):
    X = df[columns]
    y = df[target_col]

    estimator = SVR(kernel="linear")
    selector = RFE(estimator, n_features_to_select=len(columns), step=1)
    selector = selector.fit(X, y)
    df = pd.DataFrame(selector.transform(X), columns=columns)
    df[target_col] = y
    return df


if __name__ == '__main__':
    data = pd.DataFrame([[0.87, -1.34, 0.31, 0],
                         [-2.79, -0.02, -0.85, 1],
                         [-1.34, -0.48, -2.55, 0],
                         [1.92, 1.48, 0.65, 1]], columns=list('ABCy'))
    rfe(data, ['A', 'B', 'C'], 'y')
