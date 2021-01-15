import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression


def select_from_model(df, columns, target_col):
    X = df[columns]
    y = df[target_col]
    selector = SelectFromModel(estimator=LogisticRegression()).fit(X, y)
    true_list = list(selector.get_support())
    index = [i for i in range(len(true_list)) if true_list[i] == True]
    saved_columns = [columns[i] for i in index]
    result = pd.DataFrame(selector.transform(X), columns=saved_columns)
    result[target_col] = y
    return result


# if __name__ == '__main__':
#     data = pd.DataFrame([[0.87, -1.34, 0.31, 0],
#                          [-2.79, -0.02, -0.85, 1],
#                          [-1.34, -0.48, -2.55, 0],
#                          [1.92, 1.48, 0.65, 1]], columns=list('ABCy'))
#     select_from_model(data, ['A', 'B', 'C'], 'y')
