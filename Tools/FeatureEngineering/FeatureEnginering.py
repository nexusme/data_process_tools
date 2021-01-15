import pandas as pd
import numpy as np
import categorical_embedder as ce
from sklearn.preprocessing import add_dummy_feature, PolynomialFeatures, Binarizer, LabelBinarizer, LabelEncoder, \
    MaxAbsScaler, MinMaxScaler, OneHotEncoder, StandardScaler

from sklearn.feature_selection import RFE, RFECV, SelectFdr, chi2, SelectFromModel, SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from category_encoders.polynomial import PolynomialEncoder
from category_encoders.target_encoder import TargetEncoder


class GenerateFeatures(object):
    @staticmethod
    def dummy_features(df, value):
        """
            生成虚拟特征
            用一个额外的虚拟对象扩充数据集特写。
            这个对于将截取项与无法直接拟合的实现进行拟合非常有用。
        """
        add_dummy_feature(df, value=value)
        return df

    @staticmethod
    def poly_features(df, poly_num, inter_flag):
        """
            生成新的特征矩阵，包含所有多项式组合的特征，
            其次数小于或等于指定的次数学位
            例如，如果输入样本是二维的且形式为[a，b]，则二次多项式特征为[1，a，b，a^2，ab，b^2]。
        """
        poly = PolynomialFeatures(degree=poly_num, interaction_only=inter_flag)
        df = poly.fit_transform(df)
        return df


class SelectFeatures(object):
    @staticmethod
    def rfe(df, col, target_col):
        """
            递归特征消除的目标（RFE）
            针对那些特征含有权重的预测模型，RFE通过递归的方式，
            不断减少特征集的规模来选择需要的特征。
        """
        X = df[col]
        y = df[target_col]

        estimator = SVR(kernel="linear")
        selector = RFE(estimator, n_features_to_select=len(col), step=1)
        selector = selector.fit(X, y)
        df = pd.DataFrame(selector.transform(X), columns=col)
        df[target_col] = y
        return df

    @staticmethod
    def rfecv(df, col, target_col):
        """
            带交叉验证带递归特征消除的目标（RFE）
        """
        X = df[col]
        y = df[target_col]
        estimator = SVR(kernel="linear")
        selector = RFECV(estimator, step=1, cv=len(col))
        selector = selector.fit(X, y)
        data = selector.transform(X)
        # get kept columns
        true_list = list(selector.get_support())
        index = [i for i in range(len(true_list)) if true_list[i] == True]
        saved_columns = [col[i] for i in index]
        # save into dataframe
        result = pd.DataFrame(data, columns=saved_columns)
        result[target_col] = y
        return result

    @staticmethod
    def select_fdr(df, target_col):
        """
            FDR错误发现率-P值校正学习
            筛选器：为估计的错误发现率选择p值。
        """
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

    @staticmethod
    def select_from_model(df, col, target_col):
        """
            基于重要性权重选择特征的元变换器。
        """
        X = df[col]
        y = df[target_col]
        selector = SelectFromModel(estimator=LogisticRegression()).fit(X, y)
        true_list = list(selector.get_support())
        index = [i for i in range(len(true_list)) if true_list[i] == True]
        saved_columns = [col[i] for i in index]
        result = pd.DataFrame(selector.transform(X), columns=saved_columns)
        result[target_col] = y
        return result

    @staticmethod
    def select_percentile(df, target_col, keep_per):
        """
            根据最高值的百分位数选择特征分数
        """
        y = df[target_col]
        X = df.drop(target_col, axis=1)
        selector = SelectPercentile(chi2, percentile=keep_per).fit(X, y)
        true_list = list(selector.get_support())
        index = [i for i in range(len(true_list)) if true_list[i] == True]
        saved_columns = [list(X.columns)[i] for i in index]
        result = pd.DataFrame(selector.transform(X), columns=saved_columns)
        result[target_col] = y
        return result


class TransferFeatures(object):
    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def get_chi_square(df_name, ori_bins):
        chi_square_values_list = []
        for i in range(len(ori_bins) - 2):
            # print(ori_bins[i])
            chi_square_value = TransferFeatures.calculate_chi_square(df_name, ori_bins, i)
            chi_square_values_list.append(chi_square_value)
        return chi_square_values_list

    @staticmethod
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

    @staticmethod
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

        original_bins = TransferFeatures.ini_chi_merge(original_bins, df)
        # print(original_bins)
        # 开始计算最初的卡方值
        chi_square_list = TransferFeatures.get_chi_square(df, original_bins)
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
                chi_square_list[min_chi_square_index + 1] = TransferFeatures.calculate_chi_square(df, original_bins,
                                                                                                  min_chi_square_index)
            elif min_chi_square_index == len(chi_square_list) - 1:
                chi_square_list[min_chi_square_index - 1] = TransferFeatures.calculate_chi_square(df, original_bins,
                                                                                                  min_chi_square_index - 1)
            else:
                chi_square_list[min_chi_square_index - 1] = TransferFeatures.calculate_chi_square(df, original_bins,
                                                                                                  min_chi_square_index - 1)
                chi_square_list[min_chi_square_index + 1] = TransferFeatures.calculate_chi_square(df, original_bins,
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

    @staticmethod
    def chi_square_merge(df_name, y_col_name, col_name):
        """
        卡方分箱
        :param df_name: dataframe
        :param y_col_name: 作为y的列
        :param col_name 待处理待列
        :return: data_set, iv_dict
        """
        print('Chi merge is processing:')
        df_m = TransferFeatures.change_name_y(df_name, y_col_name)

        x_train, x_test, y_train, y_test = train_test_split(df_m, df_m['y'], test_size=0.3, random_state=0)

        df_new = pd.DataFrame({y_col_name: y_train})
        df_train_woe = df_new.sort_index()

        df_new_test = pd.DataFrame({y_col_name: y_test})
        df_test_woe = df_new_test.sort_index()

        df_woe_cut = pd.DataFrame()

        cols = col_name

        iv_dict = {}
        for column in cols:
            get_column_value, get_test_column, get_new_cut_woe_result, iv_record = TransferFeatures.chi_square_merge_goes(
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

    @staticmethod
    def cut_data(df_name, col, target):
        """
        等频分箱
        :param df_name: dataframe
        :param col: column
        :param target: y
        :return:
        """
        n = 10
        badsum = df_name[target].sum()
        goodsum = df_name[target].count() - badsum
        d1 = pd.concat((df_name[col], df_name[target]), axis=1)
        d1['bucket'] = pd.cut(df_name[col], n)
        d2 = d1.groupby('bucket', as_index=True)
        d3 = pd.DataFrame()
        d3['total'] = d2.count()[target]
        d3['max'] = d2.max()[col]
        d3['badsum'] = d2.sum()[target]
        d3['goodsum'] = d2.count()[target] - d3.badsum
        d3['bad_rate'] = d3['badsum'] / d3['total']
        d3['good_rate'] = d3['goodsum'] / d3['total']
        d3['total_rate'] = d3['total'] / df_name[target].count()
        woe = np.log((d3.goodsum / goodsum) / (d3.badsum / badsum))
        d3['woe'] = woe
        d3['iv'] = ((d3['goodsum'] / goodsum) - (d3['badsum'] / badsum)) * d3['woe']
        d3['total_iv'] = d3['iv'].sum()
        IV = d3['iv'].sum()
        d3['name'] = col
        cut = list(d3['max'].round(6))
        cut.insert(0, float('-inf'))
        cut[-1] = float('inf')
        print(d3
              )
        return d3, cut, IV, woe

    @staticmethod
    def optimal_binning_boundary(df_name, col, target):
        """
        决策树分箱
        :param df_name: dataframe
        :param col: column name
        :param target: y
        :return:
        """
        boundary = []
        x = df_name[col].values
        y = df_name[target].values
        clf = DecisionTreeClassifier(criterion='entropy',
                                     max_leaf_nodes=6,
                                     min_samples_leaf=0.05)
        clf.fit(x.reshape(-1, 1), y)
        n_nodes = clf.tree_.node_count
        children_left = clf.tree_.children_left
        children_right = clf.tree_.children_right
        threshold = clf.tree_.threshold
        for i in range(n_nodes):
            if children_left[i] != children_right[i]:
                boundary.append(threshold[i])
        boundary.sort()
        boundary.insert(0, float('-inf'))
        boundary.append(float('inf'))
        return boundary

    @staticmethod
    def decision_tree_feature_woe_iv(df_name, col, target):
        """
        决策树分箱 计算woe iv
        :param df_name: dataframe
        :param col: column which need to manage
        :param target: y
        :return:
        """
        boundary = TransferFeatures.optimal_binning_boundary(df_name, col, target)
        df1 = pd.concat([df_name[col], df_name[target]], axis=1)
        df1.columns = ['x', 'y']
        df1['bins'] = pd.cut(x=df_name[col], bins=boundary, right=False)
        grouped = df1.groupby('bins')['y']
        result_df = grouped.agg([('good', lambda y: (y == 0).sum()),
                                 ('bad', lambda y: (y == 1).sum()),
                                 ('total', 'count')])

        result_df['good_pct'] = result_df['good'] / result_df['good'].sum()
        result_df['bad_pct'] = result_df['bad'] / result_df['bad'].sum()
        result_df['total_pct'] = result_df['total'] / result_df['total'].sum()

        result_df['bad_rate'] = result_df['bad'] / result_df['total']

        woe = np.log(result_df['good_pct'] / result_df['bad_pct'])
        result_df['woe'] = woe
        result_df['iv'] = (result_df['good_pct'] - result_df['bad_pct']) * result_df['woe']
        iv = result_df['iv'].sum()
        print('result_df', result_df)
        print('boundary', boundary)
        print('woe', woe)
        print('iv', iv)
        return result_df, boundary, woe, iv

    @staticmethod
    def qcut_data(df_name, col, target):
        """
        等频分箱
        :param col: 操作的列
        :param target: y
        :param df_name: dataframe
        :return:
        """
        n = 5
        badsum = df_name[target].sum()
        goodsum = df_name[target].count() - badsum
        d1 = pd.concat((df_name[col], df_name[target]), axis=1)
        d1['bucket'] = pd.qcut(df_name[col], n)
        d2 = d1.groupby('bucket', as_index=True)
        d3 = pd.DataFrame()
        d3['total'] = d2.count()[target]
        d3['max'] = d2.max()[col]
        d3['badsum'] = d2.sum()[target]
        d3['goodsum'] = d2.count()[target] - d3.badsum
        d3['bad_rate'] = d3['badsum'] / d3['total']
        d3['good_rate'] = d3['goodsum'] / d3['total']
        d3['total_rate'] = d3['total'] / df_name[target].count()
        woe = np.log((d3.goodsum / goodsum) / (d3.badsum / badsum))
        d3['woe'] = woe
        d3['iv'] = ((d3['goodsum'] / goodsum) - (d3['badsum'] / badsum)) * d3['woe']
        d3['total_iv'] = d3['iv'].sum()
        IV = d3['iv'].sum()
        d3['name'] = col
        cut = list(d3['max'].round(6))
        cut.insert(0, float('-inf'))
        cut[-1] = float('inf')
        print(d3)
        print(cut)
        print(IV)
        print(woe)
        return d3, cut, IV, woe

    @staticmethod
    def binarizer(df):
        """
            根据阈值对数据进行二值化（将特征值设置为0或1）
        """
        X = df.values
        transformer = Binarizer().fit(X)  # fit does nothing.
        matrix = transformer.transform(X)
        return matrix

    @staticmethod
    def category_embedder(df, target_col):
        """
           函数逼近问题中的分类变量映射到欧氏空间
           更快的模特训练
           更少的内存消耗
           可以提供比1-hot编码更好的精度
        """
        """
            !pip install tensorflow_addons==0.8.3
            !pip install tqdm==4.41.1
            !pip install keras==2.3.1
            !pip install tensorflow==2.2.0
        """

        X = df.drop([target_col], axis=1)
        y = df[target_col]

        # ce.get_embedding_info identifies the categorical variables
        # of unique values and embedding size and returns a dictionary
        embedding_info = ce.get_embedding_info(X)

        X_encoded, encoders = ce.get_label_encoded_data(X)

        # splitting the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y)

        embeddings = ce.get_embeddings(X_train, y_train, categorical_embedding_info=embedding_info,
                                       is_classification=True,
                                       epochs=100, batch_size=256)

        # convert it to dataframe for easy readibility
        # dfs = ce.get_embeddings_in_dataframe(embeddings=embeddings, encoders=encoders)
        data = ce.fit_transform(X, embeddings=embeddings, encoders=encoders, drop_categorical_vars=True)
        print(data)
        return data

    @staticmethod
    def count_encoding(df, col):
        """
            •将分类变量替换为序列集中的计数
            •适用于线性和非线性算法
            •对异常值敏感
            •可添加对数转换，与计数配合良好
            •将看不见的变量替换为`1`
            •可能产生冲突：相同的编码，不同的变量
        """
        map_dict = dict(Counter(df[col]))
        df[col] = df[col].apply(lambda x: map_dict[x])
        return df

    @staticmethod
    def label_binarizer(df, col):
        """
            以一对一的方式对标签进行二值化 适用于多分类
        """
        lb = LabelBinarizer()
        label_list = lb.fit_transform(df[col]).flatten()
        df[col] = label_list
        return df

    @staticmethod
    def label_encoding(df_name, col):
        """
        标签编码 特征存在内在顺序 (ordinal feature)
        对于一个有m个category的特征，经过label encoding以后，
        每个category会映射到0到m-1之间的一个数。label encoding适用于ordinal feature （特征存在内在顺序）。
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

    @staticmethod
    def max_abs_scale(df):
        """
            按最大绝对值缩放每个特征。
            该估计器分别对每个特征进行缩放和平移，使得训练集中每个特征的最大绝对值为1.0。
            它不会移动/集中数据，因此不会破坏任何稀疏性。
        """
        X = df
        transformer = MaxAbsScaler().fit(X)
        df = pd.DataFrame(transformer.transform(X), columns=df.columns)
        return df

    @staticmethod
    def min_max(df_name, col):
        """
        归一化
        :param col: 待处理列名
        :param df_name: dataframe
        """
        data = df_name[col]
        min_max_scaler = MinMaxScaler()
        min_max_param = min_max_scaler.fit_transform(data.values.reshape(-1, 1))
        df_name[col] = min_max_scaler.fit_transform(df_name[col].values.reshape(-1, 1), min_max_param)
        return df_name

    @staticmethod
    def nan_encoding(df, col):
        """
            •为NaN值提供显式编码，而不是忽略
            •NaN值可以保存信息
            •小心避免过拟合
            •仅当训练和测试集中的NaN值是由同一原因引起时，或当本地验证证明其保持信号时使用
        """
        df[col] = df[col].apply(lambda x: 1 if np.isnan(x) == 1 else 0)
        return df

    @staticmethod
    def one_hot(df_name, cols):
        """
            特征无内在顺序，category数量 < 4
        """
        df = df_name
        for col in cols:
            df = TransferFeatures.one_hot_single(df, col)
        return df

    @staticmethod
    def one_hot_single(df_name, col):
        """
            一位有效编码，主要是采用N位状态寄存器来对N个状态进行编码
        """
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

    @staticmethod
    def poly_encoding(cols, train_set, train_y, test_set):
        """
            •编码分类变量之间的相互作用
            •没有交互作用的线性算法无法解决异或问题
            •多项式核*可以*解异或
            •分解特征空间：使用FS、散列和/或VW
        """
        poly = PolynomialEncoder(cols=cols, handle_unknown='value', handle_missing='value').fit(train_set, train_y)
        poly_tr = poly.transform(train_set, train_set)
        poly_tst = poly.transform(test_set)

        return poly_tr, poly_tst

    @staticmethod
    def standard_scale(df_name, col):
        """
        标准化 通过去除均值并缩放到单位方差来标准化特征
        :param col: 待处理列名
        :param df_name: dataframe
        """
        data = df_name[col]
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        scale_param = scaler.fit(data.values.reshape(-1, 1))
        df_name[col] = scaler.fit_transform(df_name[col].values.reshape(-1, 1), scale_param)
        return df_name

    @staticmethod
    def target_encoder(cols, train_set, train_y, test_set):
        """
            特征无内在顺序，category数量 > 4
            Target encoding 采用 target mean value （among each category） 来给categorical feature做编码。
            handle_unknown 和 handle_missing 被设定为 'value'
            在目标编码中，handle_unknown 和 handle_missing 仅接受 ‘error’, ‘return_nan’ 及 ‘value’ 设定
            两者的默认值均为 ‘value’, 即对未知类别或缺失值填充训练集的因变量平均值
        """
        encoder = TargetEncoder(cols=cols, handle_unknown='value', handle_missing='value').fit(train_set, train_y)
        encoded_train = encoder.transform(train_set)  # 转换训练集
        encoded_test = encoder.transform(test_set)  # 转换测试集

        return encoded_train, encoded_test

    @staticmethod
    def z_score(df_name, col):
        """
        z-score编码
        Z值的量代表着原始分数和母体平均值之间的距离，是以标准差为单位计算。
        在原始分数低于平均值时Z则为负数，反之则为正数。
        :param col: 待处理列名
        :param df_name: dataframe
        """
        data = df_name[col]
        print(data)
        length = len(data)
        total = sum(data)
        ave = float(total) / length
        list_data = list(data)
        tmp_sum = sum([pow(list_data[i] - ave, 2) for i in range(length)])
        tmp_sum = pow(float(tmp_sum) / length, 0.5)
        for i in range(length):
            list_data[i] = (list_data[i] - ave) / tmp_sum
        df_name[col] = pd.DataFrame(list_data)
        return df_name
