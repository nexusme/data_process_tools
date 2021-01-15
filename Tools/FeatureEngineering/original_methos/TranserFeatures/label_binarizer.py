from sklearn import preprocessing
import pandas as pd


def binarizer(df, col):
    lb = preprocessing.LabelBinarizer()
    label_list = lb.fit_transform(df[col]).flatten()
    df[col] = label_list
    return df


if __name__ == '__main__':
    df_1 = pd.DataFrame([['dog', 2, 1, 0, 'T'],
                         ['cat', 4, 6, 1, 'F'],
                         ['wolf', 9, 6, 5, 'F'],
                         ['rabbit', 3, 7, 4, 'T']],
                        columns=list('ABCDy'))
    print(df_1)
    d = binarizer(df_1, 'y')
    print(d)
