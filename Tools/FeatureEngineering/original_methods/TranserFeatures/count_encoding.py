import pandas as pd
import numpy as np
from collections import Counter


def count_encoding(df, col):
    """
        Replace categorical variables with their count in the train set
    """
    map_dict = dict(Counter(df[col]))
    df[col] = df[col].apply(lambda x: map_dict[x])
    return df


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
