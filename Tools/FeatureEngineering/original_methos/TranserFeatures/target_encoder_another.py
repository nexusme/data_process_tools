import pandas as pd
import numpy as np
from category_encoders.target_encoder import TargetEncoder


def target_encoder(cols, train_set, train_y, test_set):
    # handle_unknown 和 handle_missing 被设定为 'value'
    # 在目标编码中，handle_unknown 和 handle_missing 仅接受 ‘error’, ‘return_nan’ 及 ‘value’ 设定
    # 两者的默认值均为 ‘value’, 即对未知类别或缺失值填充训练集的因变量平均值
    encoder = TargetEncoder(cols=cols, handle_unknown='value', handle_missing='value').fit(train_set, train_y)
    encoded_train = encoder.transform(train_set)  # 转换训练集
    encoded_test = encoder.transform(test_set)  # 转换测试集

    return encoded_train, encoded_test


if __name__ == "__main__":
    # 随机生成一些训练集
    t_set = pd.DataFrame(np.array([['male', 10], ['female', 20], ['male', 10],
                                   ['female', 20], ['female', 15]]),
                         columns=['Sex', 'Type'])
    t_y = np.array([False, True, True, False, False])

    # 随机生成一些测试集, 并有意让其包含未在训练集出现过的类别与缺失值
    tst_set = pd.DataFrame(np.array([['female', 20], ['male', 20], ['others', 15],
                                     ['male', 20], ['female', 40], ['male', 25]]),
                           columns=['Sex', 'Type'])
    tst_set.loc[4, 'Type'] = np.nan

    print(t_set)
    print(tst_set)

    trans_train, trans_test = target_encoder(['Sex', 'Type'], t_set, t_y, tst_set)
    print(trans_train)
    print(trans_test)
