from sklearn.preprocessing import MaxAbsScaler
import pandas as pd


def max_abs_scale(df):
    X = df
    transformer = MaxAbsScaler().fit(X)
    df = pd.DataFrame(transformer.transform(X), columns=df.columns)
    return df
    # print(df)

#
# if __name__ == '__main__':
#     df_1 = pd.DataFrame([[5, 2, 1, 0],
#                          [7, 4, 6, 1],
#                          [3, 9, 6, 5],
#                          [1, 3, 7, 4]],
#                         columns=list('ABCD'))
#     print(df_1)
#     d = max_abs_scale(df_1)
#     print(d)
