import pandas as pd
import numpy as np


def data_sampling(df):
    column = 'A'
    print(df[column].sample(n=len(df), random_state=1))
    return df[column].sample(n=len(df), random_state=1)

# if __name__ == '__main__':
#     df_1 = pd.DataFrame([[np.nan, 2, np.nan, 0],
#                          [3, 4, np.nan, 1],
#                          [2, np.nan, np.nan, 5],
#                          [np.nan, 3, np.nan, 4]],
#                         columns=list('ABCD'))
#     d = data_sampling(df_1,'D')
