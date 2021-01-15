from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def change_name_y(df_name, col_name):
    """
    将特征改名为y
    :param df_name: dataframe
    :param col_name: 要变更为y的列
    :return:
    """
    df_m = df_name
    cols = list(df_m)
    cols.insert(0, cols.pop(cols.index(col_name)))
    df_m = df_m.loc[:, cols]
    df_m.rename(columns={col_name: 'y'}, inplace=True)
    return df_m


def chi_bins(df_name, col, target, confidence=3.841, bins=20):  # 设定自由度为1，卡方阈值为3.841，最大分箱数20
    """
    卡方分箱
    :param df_name:  dataframe
    :param col: column which waiting for cut
    :param target: y
    :param confidence:
    :param bins: bin nums
    :return:
    """
    total = df_name[target].count()  # 计算总样本数
    bad = df_name[target].sum()  # 计算坏样本总数
    good = total - bad  # 计算好样本总数
    total_bin = df_name.groupby([col])[target].count()  # 计算每个箱体总样本数
    total_bin_table = pd.DataFrame({'total': total_bin})  # 创建一个数据框保存结果
    bad_bin = df_name.groupby([col])[target].sum()  # 计算每个箱体的坏样本数
    bad_bin_table = pd.DataFrame({'bad': bad_bin})  # 创建一个数据框保存结果
    regroup = pd.merge(total_bin_table, bad_bin_table, left_index=True, right_index=True,
                       how='inner')  # 组合total_bin 和 bad_bin
    regroup.reset_index(inplace=True)
    regroup['good'] = regroup['total'] - regroup['bad']  # 计算每个箱体的好样本数
    regroup = regroup.drop(['total'], axis=1)  # 删除total
    np_regroup = np.array(regroup)  # 将regroup转为numpy

    # 处理连续没有正样本和负样本的区间，进行合并，以免卡方报错
    i = 0
    while (i <= np_regroup.shape[0] - 2):
        if ((np_regroup[i, 1] == 0 and np_regroup[i + 1, 1] == 0) or (
                np_regroup[i, 2] == 0 and np_regroup[i + 1, 2] == 0)):
            np_regroup[i, 1] = np_regroup[i, 1] + np_regroup[i + 1, 1]  # 正样本
            np_regroup[i, 2] = np_regroup[i, 2] + np_regroup[i + 1, 2]  # 负样本
            np_regroup[i, 0] = np_regroup[i + 1, 0]
            np_regroup = np.delete(np_regroup, i + 1, 0)
            i = i - 1
        i = i + 1

    # 对相邻两个区间的值进行卡方计算
    chi_table = np.array([])  # 创建一个数组保存相邻两个区间的卡方值
    for i in np.arange(np_regroup.shape[0] - 1):
        chi = ((np_regroup[i, 1] * np_regroup[i + 1, 2] - np_regroup[i, 2] * np_regroup[i + 1, 1]) ** 2 * \
               (np_regroup[i, 1] + np_regroup[i, 2] + np_regroup[i + 1, 1] + np_regroup[i + 1, 2])) / \
              ((np_regroup[i, 1] + np_regroup[i, 2]) * (np_regroup[i + 1, 1] + np_regroup[i + 1, 2]) * \
               (np_regroup[i, 1] + np_regroup[i + 1, 1]) * (np_regroup[i, 2] + np_regroup[i + 1, 2]))
        chi_table = np.append(chi_table, chi)

    # 将卡方值最小的两个区间进行合并
    while (1):  # 除非设置break，否则会一直循环下去
        if (len(chi_table) <= (bins - 1) or min(chi_table) >= confidence):
            break  # 当chi_table的值个数小于等于（箱体数-1) 或 最小的卡方值大于等于卡方阈值时，循环停止
        chi_min_index = np.where(chi_table == min(chi_table))[0]  # 这个地方要注意
        np_regroup[chi_min_index, 1] = np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]
        np_regroup[chi_min_index, 2] = np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]
        np_regroup[chi_min_index, 0] = np_regroup[chi_min_index + 1, 0]
        np_regroup = np.delete(np_regroup, chi_min_index + 1, axis=0)

        print(chi_min_index, np_regroup.shape[0] - 1)

        if (chi_min_index == np_regroup.shape[0] - 1):  # 当卡方最小值是最后两个区间时，计算当前区间和前一个区间的卡方值并替换
            chi_table[chi_min_index - 1] = ((np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] -
                                             np_regroup[chi_min_index - 1, 2] \
                                             * np_regroup[chi_min_index, 1]) ** 2 * (
                                                    np_regroup[chi_min_index - 1, 1] + np_regroup[
                                                chi_min_index - 1, 2] \
                                                    + np_regroup[chi_min_index, 1] + np_regroup[
                                                        chi_min_index, 2])) / ((np_regroup[chi_min_index - 1, 1] + \
                                                                                np_regroup[
                                                                                    chi_min_index - 1, 2]) * (
                                                                                       np_regroup[
                                                                                           chi_min_index, 1] +
                                                                                       np_regroup[
                                                                                           chi_min_index, 2]) * \
                                                                               (np_regroup[chi_min_index - 1, 1] +
                                                                                np_regroup[chi_min_index, 1]) * (
                                                                                       np_regroup[
                                                                                           chi_min_index - 1, 2] + \
                                                                                       np_regroup[
                                                                                           chi_min_index, 2]))
            chi_table = np.delete(chi_table, chi_min_index, axis=0)  # 删除替换前的卡方值
        else:
            # 计算合并后当前区间和前一个区间的卡方值并替换
            chi_table[chi_min_index - 1] = ((np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] -
                                             np_regroup[chi_min_index - 1, 2] \
                                             * np_regroup[chi_min_index, 1]) ** 2 * (
                                                    np_regroup[chi_min_index - 1, 1] + np_regroup[
                                                chi_min_index - 1, 2] \
                                                    + np_regroup[chi_min_index, 1] + np_regroup[
                                                        chi_min_index, 2])) / ((np_regroup[chi_min_index - 1, 1] + \
                                                                                np_regroup[
                                                                                    chi_min_index - 1, 2]) * (
                                                                                       np_regroup[
                                                                                           chi_min_index, 1] +
                                                                                       np_regroup[
                                                                                           chi_min_index, 2]) * \
                                                                               (np_regroup[chi_min_index - 1, 1] +
                                                                                np_regroup[chi_min_index, 1]) * (
                                                                                       np_regroup[
                                                                                           chi_min_index - 1, 2] + \
                                                                                       np_regroup[
                                                                                           chi_min_index, 2]))

            # 计算合并后当前区间和后一个区间的卡方值并替换
            chi_table[chi_min_index] = ((np_regroup[chi_min_index, 1] * np_regroup[chi_min_index + 1, 2] -
                                         np_regroup[
                                             chi_min_index, 2] \
                                         * np_regroup[chi_min_index + 1, 1]) ** 2 * (
                                                np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2] \
                                                + np_regroup[chi_min_index + 1, 1] + np_regroup[
                                                    chi_min_index + 1, 2])) / ((np_regroup[chi_min_index, 1] + \
                                                                                np_regroup[chi_min_index, 2]) * (
                                                                                       np_regroup[
                                                                                           chi_min_index + 1, 1] +
                                                                                       np_regroup[
                                                                                           chi_min_index + 1, 2]) * \
                                                                               (np_regroup[chi_min_index, 1] +
                                                                                np_regroup[
                                                                                    chi_min_index + 1, 1]) * (
                                                                                       np_regroup[
                                                                                           chi_min_index, 2] + \
                                                                                       np_regroup[
                                                                                           chi_min_index + 1, 2]))
            chi_table = np.delete(chi_table, chi_min_index + 1, axis=0)  # 删除替换前的卡方值

    # 将结果保存为一个数据框
    result_data = pd.DataFrame()
    result_data['col'] = [col] * np_regroup.shape[0]  # 结果第一列为变量名
    list_temp = []  # 创建一个空白的分组列
    for i in np.arange(np_regroup.shape[0]):
        if i == 0:  # 当为第一个箱体时
            x = '0' + ',' + str(np_regroup[i, 0])
        elif i == np_regroup.shape[0] - 1:  # 当为最后一个箱体时
            x = str(np_regroup[i - 1, 0]) + '+'
        else:
            x = str(np_regroup[i - 1, 0]) + ',' + str(np_regroup[i, 0])
        list_temp.append(x)
    result_data['bin'] = list_temp
    result_data['bad'] = np_regroup[:, 1]
    result_data['good'] = np_regroup[:, 2]
    result_data['bad_rate'] = result_data['bad'] / total  # 计算每个箱体坏样本所占总样本比例
    result_data['badattr'] = result_data['bad'] / bad  # 计算每个箱体坏样本所占坏样本总数的比例
    result_data['goodattr'] = result_data['good'] / good  # 计算每个箱体好样本所占好样本总数的比例
    result_data['woe'] = np.log(result_data['goodattr'] / result_data['badattr'])  # 计算每个箱体的woe值
    iv = ((result_data['goodattr'] - result_data['badattr']) * result_data['woe']).sum()  # 计算每个变量的iv值
    print('分箱结果:')
    print(result_data)
    print('IV值为:')
    print(iv)
    return result_data, iv


def calculate_chi_square(df_name, bins, index):
    box1 = df_name[(df_name['X'] >= bins[index]) & (df_name['X'] < bins[index + 1])]
    box2 = df_name[(df_name['X'] >= bins[index + 1]) & (df_name['X'] < bins[index + 2])]

    # print(len(box1), len(box2), index)
    a11 = len(box1[box1["Y"] == 0])
    a12 = len(box1[box1['Y'] == 1])
    a21 = len(box2[box2['Y'] == 0])
    a22 = len(box2[box2['Y'] == 1])

    a_list = [a11, a12, a21, a22]

    r1 = a11 + a12
    r2 = a21 + a22
    n = r1 + r2
    c1 = a11 + a21
    c2 = a12 + a22

    e11 = r1 * c1 / n
    e12 = r1 * c2 / n
    e21 = r2 * c1 / n
    e22 = r2 * c2 / n

    e_list = [e11, e12, e21, e22]

    chi_square = 0
    try:
        for k in range(len(a_list)):
            chi_square += (a_list[k] - e_list[k]) ** 2 / e_list[k]
    except:
        # print(a_list, e_list, bins[index], r1, r2, c1, c2)
        raise
    return chi_square


def get_chi_square(df_name, ori_bins):
    chi_square_values_list = []
    for i in range(len(ori_bins) - 2):
        # print(ori_bins[i])
        chi_square_value = calculate_chi_square(df_name, ori_bins, i)
        chi_square_values_list.append(chi_square_value)
    return chi_square_values_list


def ini_chi_merge(bins, df_name):
    i = 0

    while i < len(bins) - 2:
        df_adjacent = df_name[(df_name['X'] >= bins[i]) & (df_name['X'] < bins[i + 2])]
        if len(df_adjacent[df_adjacent["Y"] == 0]) == 0 or len(df_adjacent[df_adjacent["Y"] == 1]) == 0:
            del bins[i + 1]
            i -= 1
        i += 1
    # print(bins)
    return bins


def chi_square_merge_goes(X_test_column, Y_test, Y_train, X_column, col_name):
    print(col_name + ' is processing...')
    # Y_train = 1 响应 Y_train = 0 未响应
    # X_column待分箱变量
    # 卡方分箱阈值
    THRESHOLD = 10000
    # 分箱数目限制
    LIMIT_NUM = 6

    df = pd.DataFrame({"Y": Y_train, "X": X_column})
    df = df.sort_values("X")

    df_test = pd.DataFrame({"Y": Y_test, "X": X_test_column})
    df_test = df_test.sort_values("X")
    # print('test数据\n', df_test)

    # print(df)

    original_bins = sorted(set(df["X"].values))

    total = df.Y.count()  # 计算总样本数
    bad = df.Y.sum()  # 计算坏样本总数
    good = total - bad  # 计算好样本总数
    total_bin = df.groupby(['X'])['Y'].count()  # 计算每个箱体总样本数
    total_bin_table = pd.DataFrame({'total': total_bin})  # 创建一个数据框保存结果
    bad_bin = df.groupby(['X'])['Y'].sum()  # 计算每个箱体的坏样本数
    bad_bin_table = pd.DataFrame({'bad': bad_bin})  # 创建一个数据框保存结果
    regroup = pd.merge(total_bin_table, bad_bin_table, left_index=True, right_index=True,
                       how='inner')  # 组合total_bin 和 bad_bin
    regroup.reset_index(inplace=True)
    regroup['good'] = regroup['total'] - regroup['bad']  # 计算每个箱体的好样本数
    regroup = regroup.drop(['total'], axis=1)  # 删除total
    # print(regroup)

    np_regroup = np.array(regroup)  # 将regroup转为numpy
    # print(np_regroup)

    original_bins.append(np.inf)

    # 预处理 合并全为0/1的区间

    original_bins = ini_chi_merge(original_bins, df)
    # print(original_bins)
    # 开始计算最初的卡方值
    chi_square_list = get_chi_square(df, original_bins)
    # print(chi_square_list)
    # print(chi_square_list)
    # 开始合并
    while 1:
        # print('Current chi merge box is: ', len(original_bins) - 1)

        min_chi_square = min(chi_square_list)

        min_chi_square_index = chi_square_list.index(min_chi_square)
        # print('The min index is: ', min_chi_square_index)
        # print('Original bin is: ', original_bins)
        # print('The length of original bins is: ', len(original_bins))
        del original_bins[min_chi_square_index + 1]
        if min_chi_square_index == 0:
            chi_square_list[min_chi_square_index + 1] = calculate_chi_square(df, original_bins,
                                                                             min_chi_square_index)
        elif min_chi_square_index == len(chi_square_list) - 1:
            chi_square_list[min_chi_square_index - 1] = calculate_chi_square(df, original_bins,
                                                                             min_chi_square_index - 1)
        else:
            chi_square_list[min_chi_square_index - 1] = calculate_chi_square(df, original_bins,
                                                                             min_chi_square_index - 1)
            chi_square_list[min_chi_square_index + 1] = calculate_chi_square(df, original_bins,
                                                                             min_chi_square_index)
        del chi_square_list[min_chi_square_index]

        if min_chi_square > THRESHOLD or len(original_bins) <= LIMIT_NUM:
            break

    result_data = pd.DataFrame()
    list_temp = []  # 创建一个空白的分组列
    list_bad_num = []
    list_good_num = []
    bad_attr = []
    good_attr = []
    woe = []
    i = 0
    iv_fine = 0
    feature_series = pd.Series(df['X'])
    test_feature_series = pd.Series(df_test['X'])
    iv_dict = {}
    # print(feature_series)
    while i < len(original_bins) - 1:
        list_temp.append(str(original_bins[i]) + ',' + str(original_bins[i + 1]))
        new_cut = regroup[(regroup['X'] >= original_bins[i]) & (regroup['X'] < original_bins[i + 1])]
        # print('new cut\n', new_cut)
        # list_num.append(num)
        bad_num = new_cut['bad'].sum()
        if bad_num == 0:
            bad_num = 1
        list_bad_num.append(str(bad_num))

        good_num = new_cut['good'].sum()
        if good_num == 0:
            good_num = 1
        list_good_num.append(str(good_num))

        # bad_rate.append(str(bad_num / total))
        bad_attr.append(str(bad_num / bad))
        badatt = bad_num / bad
        good_attr.append(str(good_num / good))
        goodatt = good_num / good
        woe_value = np.log(goodatt / badatt)

        feature_series[(feature_series >= original_bins[i]) & (feature_series < original_bins[i + 1])] = woe_value

        test_feature_series[
            (test_feature_series >= original_bins[i]) & (test_feature_series < original_bins[i + 1])] = woe_value
        woe.append(str(woe_value))
        minus_value = goodatt - badatt
        iv_1 = minus_value * woe_value
        # print('iv_1', iv_1)
        iv_fine = iv_fine + iv_1
        # print('iv_fine', iv_fine)
        i += 1
    # print(feature_series)
    result_data['bin'] = list_temp
    result_data['bad'] = list_bad_num
    result_data['good'] = list_good_num
    # result_data['bad_rate'] = bad_rate  # 计算每个箱体坏样本所占总样本比例
    result_data['badattr'] = bad_attr  # 计算每个箱体坏样本所占坏样本总数的比例
    result_data['goodattr'] = good_attr  # 计算每个箱体好样本所占好样本总数的比例
    result_data['woe'] = woe  # 计算每个箱体的woe值
    # iv =  # 计算每个变量的iv值
    # print('分箱结果:')
    # print(result_data)
    result_data_woe = result_data[['bin', 'woe']]
    # print('woe结果:')
    # print(result_data_woe)

    final_pd = pd.DataFrame({col_name: df['Y'], col_name: feature_series})
    final_test_pd = pd.DataFrame({col_name: df_test['Y'], col_name: test_feature_series})
    final_pd = final_pd.sort_index()
    final_test_pd = final_test_pd.sort_index()

    # print("训练集woe编码结果\n", final_pd[col_name])
    # print("测试集woe编码结果\n", final_test_pd[col_name])

    list_new_woe = [woe]
    new_cut_woe_result = pd.DataFrame({'feature_name': col_name, 'bins_woe': list_new_woe})
    # print(new_cut_woe_result)

    return final_pd[col_name], final_test_pd[col_name], new_cut_woe_result, iv_fine


def chi_square_merge(df_name, y_col_name, col_name):
    """
    卡方分箱
    :param df_name: dataframe
    :param y_col_name: 作为y的列
    :param col_name 待处理待列
    :return: data_set, iv_dict
    """
    print('Chi merge is processing:')
    df_m = change_name_y(df_name, y_col_name)

    x_train, x_test, y_train, y_test = train_test_split(df_m, df_m['y'], test_size=0.3, random_state=0)

    df_new = pd.DataFrame({y_col_name: y_train})
    df_train_woe = df_new.sort_index()

    df_new_test = pd.DataFrame({y_col_name: y_test})
    df_test_woe = df_new_test.sort_index()

    df_woe_cut = pd.DataFrame()

    cols = col_name

    iv_dict = {}
    for column in cols:
        get_column_value, get_test_column, get_new_cut_woe_result, iv_record = chi_square_merge_goes(
            x_test[column].values,
            y_test,
            y_train,
            x_train[column].values,
            col_name=column)
        iv_dict[column] = iv_record
        df_train_woe[column] = get_column_value
        df_test_woe[column] = get_test_column
        df_woe_cut = df_woe_cut.append(get_new_cut_woe_result)
    # print('train\n', df_train_woe)
    # print('test\n', df_test_woe)
    # print('woe_cut\n', df_woe_cut)
    # print('iv', iv_dict)
    x_train_sort = x_train.sort_index()
    x_test_sort = x_test.sort_index()

    for column in cols:
        x_train_sort[column] = df_train_woe[column]
        x_test_sort[column] = df_test_woe[column]

    data_set = pd.concat([x_train_sort, x_test_sort])

    # return x_train_sort, x_test_sort, df_woe_cut, iv_dict
    return data_set, iv_dict
