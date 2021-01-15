import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def one_hot(df_name, cols):
    df = df_name
    for col in cols:
        df = one_hot_single(df, col)
    return df


def one_hot_single(df_name, col):
    data = df_name[col]
    values = np.array(data)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    one_hot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    one_hot_encoder = one_hot_encoder.fit_transform(integer_encoded)
    col_name = []

    for i in range(0, one_hot_encoder.shape[1]):
        col_name += [col + str(i)]

    df_f = pd.concat([df_name, pd.DataFrame(one_hot_encoder, columns=col_name)], axis=1)
    df_f = df_f.drop([col], axis=1)
    return df_f

# if __name__ == '__main__':
#     df_1 = pd.DataFrame([['dog', 2, 1, 0],
#                          ['cat', 4, 6, 1],
#                          ['wolf', 9, 6, 5],
#                          ['rabbit', 3, 7, 4]],
#                         columns=list('ABCD'))
#     print(df_1)
#     d = one_hot(df_1, 'A')
#     print(d)
