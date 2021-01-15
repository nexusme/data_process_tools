import pandas as pd
from sklearn.feature_selection import SelectFdr, chi2


def select_fdr(df, target_col):
    y = df[target_col]
    X = df.drop(target_col, axis=1)
    selector = SelectFdr(chi2, alpha=0.01).fit(X, y)
    true_list = list(selector.get_support())
    index = [i for i in range(len(true_list)) if true_list[i] == True]
    if len(index) == 0:
        print('No features were selected: either the data is too noisy or the selection test too strict.')
        return df
    else:
        saved_columns = [list(X.columns)[i] for i in index]
        result = pd.DataFrame(selector.transform(X), columns=saved_columns)
        result[target_col] = y
    return result


# if __name__ == '__main__':
#     data = pd.DataFrame([[0.87, 1.34, 0.31, 0],
#                          [2.79, 0.02, 0.85, 1],
#                          [1.34, 0.48, 2.55, 0],
#                          [1.92, 1.48, 0.65, 1]], columns=list('ABCy'))
#     select_fdr(data, 'y')
