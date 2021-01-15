from sklearn.preprocessing import LabelEncoder
import pandas as pd


def label_encoding(df_name, col):
    """
    标签编码
    :param col: 待处理列名
    :param df_name: dataframe
    """
    tags = df_name[col]
    le = LabelEncoder()
    le = le.fit(tags)
    label = le.transform(tags)
    # reverse = le.inverse_transform(label)
    df_name[col] = label
    return df_name

# if __name__ == '__main__':
#     df_1 = pd.DataFrame([['dog', 2, 1, 0, 'T'],
#                          ['dog', 4, 6, 1, 'F'],
#                          ['wolf', 9, 6, 5, 'F'],
#                          ['rabbit', 3, 7, 4, 'T'],
#                          ['dog', 4, 6, 1, 'F'],
#                          ['dog', 4, 6, 1, 'F'],
#                          ['wolf', 9, 6, 5, 'F'],
#                          ['rabbit', 3, 7, 4, 'T'],
#                          ['pig', 3, 9, 4, 'T']
#                          ],
#                         columns=list('ABCDy'))
#     print(df_1)
#     d = count_encoding(df_1, "A")
#     print(d)
