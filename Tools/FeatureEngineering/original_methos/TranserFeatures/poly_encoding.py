import pandas as pd
import numpy as np
from category_encoders.polynomial import PolynomialEncoder


def poly_encoding(cols, train_set, train_y, test_set):
    poly = PolynomialEncoder(cols=cols, handle_unknown='value', handle_missing='value').fit(train_set, train_y)
    poly_tr = poly.transform(train_set, train_set)
    poly_tst = poly.transform(test_set)

    return poly_tr, poly_tst


# if __name__ == "__main__":
#     t_set = pd.DataFrame(np.array([['male', 10], ['female', 20], ['male', 10],
#                                    ['female', 20], ['female', 15]]),
#                          columns=['Sex', 'Type'])
#     t_y = np.array([False, True, True, False, False])
#
#     tst_set = pd.DataFrame(np.array([['female', 20], ['male', 20], ['others', 15],
#                                      ['male', 20], ['female', 40], ['male', 25]]),
#                            columns=['Sex', 'Type'])
#
#     print(t_set)
#     print(tst_set)
#
#     trans_train, trans_test = poly_encoding(['Sex', 'Type'], t_set, t_y, tst_set)
#     print(trans_train)
#     print(trans_test)
