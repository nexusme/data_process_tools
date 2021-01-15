import pandas as pd
import numpy as np
from fancyimpute import KNN
from sklearn.ensemble import RandomForestRegressor


class DataProcess(object):
    @staticmethod
    def data_filtering(df, col, operation, value):
        """
            filter data onr col
        """
        if operation == 'dropna':
            data = df[col].dropna()

        elif operation == '==':
            data = df[df[col] == value]

        elif operation == '>=':
            data = df[df[col] >= value]

        elif operation == '>':
            data = df[df[col] > value]

        elif operation == '<=':
            data = df[df[col] <= value]

        elif operation == '<':
            data = df[df[col] < value]
        elif operation == '!=':
            data = df[df[col] != value]

        return data

    @staticmethod
    def data_sampling(df, col):
        """
            sampling data by one col
        """
        column = col
        print(df[column].sample(n=len(df), random_state=1))
        return df[column].sample(n=len(df), random_state=1)

    @staticmethod
    def drop_columns(df, col):
        """
        删除指定列
        :param df:
        :param col:
        :return:

        """

        df = df.drop(col, axis=1)
        return df

    @staticmethod
    def drop_duplicate(df, col):
        """
        去重特定列中的重复值
        :param df:
        :param col:
        :return:
        """
        df = df.drop_duplicates(subset=col)
        return df

    @staticmethod
    def feature_corr(df):
        """
        特征相关性
        :param df:
        """
        df.corr(method='pearson')
        print(df.corr(method='pearson'))
        return df.corr(method='pearson')

    @staticmethod
    def fillna_by_back(df, col):
        """
        缺失值填充： 用后一个数据进行填充
        :param df:
        :param col:
        :return:
        """
        df[col] = df[col].fillna(method='bfill')
        return df

    @staticmethod
    def fillna_by_forward(df, col):
        """
        缺失值填充： 用前一个数据进行填充
        :param df:
        :param col:
        :return:
        """
        df[col] = df[col].fillna(method='pad')
        return df

    @staticmethod
    def fillna_by_inter(df, col):
        """
        插值法填充
        :param df:
        :param col:
        :return:
        """
        df[col] = df[col].interpolate()
        return df

    @staticmethod
    def fillna_by_knn(df):
        """
        KNN填充
        :param df:
        :return:
        """
        data = pd.DataFrame(KNN(k=6).fit_transform(df))
        data.columns = df.columns
        return data

    @staticmethod
    def fillna_by_mean(df, col):
        """
        缺失值填充： 均值填充
        :param df:
        :param col:
        :return:
        """
        df[col] = df[col].fillna(df[col].mean())
        return df

    @staticmethod
    def fillna_by_median(df, col):
        """
        缺失值填充： 中位数填充
        :param df:
        :param col:
        :return:
        """
        df[col] = df[col].fillna(df[col].median())
        return df

    @staticmethod
    def fillna_by_mode(df, col):
        """
        缺失值填充： 使用指定数字填充
        :param df:
        :param col:
        :return:
        """
        df[col] = df[col].fillna(df[col].mode())
        return df

    @staticmethod
    def fillna_by_rf(df, col):
        """
        随机森林填补缺失值
        :param df:  dataframe
        :param col:  column to fill
        :return:
        """
        # 将需要处理的放到首列
        df_m = df
        df_move = df_m[col]
        df_m = df_m.drop(col, axis=1)
        df_m.insert(0, col, df_move)

        # 把数值型特征放入随机森林里
        missing_part_df = df_m

        known_part = missing_part_df[missing_part_df[col].notnull()].fillna(0).values

        unknown_part = missing_part_df[missing_part_df[col].isnull()].fillna(0).values
        y = known_part[:, 0]  # y 第一列数据
        x = known_part[:, 1:]  # x 是特征属性值，后面几列
        rfr = RandomForestRegressor(random_state=0, n_estimators=15, max_depth=10, n_jobs=-1)
        # 根据已有数据去拟合随机森林模型
        rfr.fit(x, y)
        # 预测缺失值
        predicted_results = rfr.predict(unknown_part[:, 1:])
        predicted_results = [int(x) for x in predicted_results]
        # print(predicted_results)
        # 填补缺失值
        df.loc[(df[col].isnull()), col] = predicted_results

        # print('Missing data has been filled up.')
        print(df)
        return df

    @staticmethod
    def lower_upper_case(df, col, type_r):
        """
            大小写转换
        """
        if type_r == 'upper':
            df[col] = df[col].str.upper()
        elif type_r == 'lower':
            df[col] = df[col].str.lower()
        return df

    @staticmethod
    def outliers_proc(data, col, scale=3):
        """
        箱线图异常值剔除，默认用 box_plot（scale=3）进行清洗
        :param data: 接收 pandas 数据格式
        :param col: pandas 列名
        :param scale: 尺度
        :return:
        """
        data_m = data.copy()
        data_series = data_m[col]

        iqr = scale * (data_series.quantile(0.75) - data_series.quantile(0.25))
        val_low = data_series.quantile(0.25) - iqr
        val_up = data_series.quantile(0.75) + iqr
        rule_low = (data_series < val_low)
        rule_up = (data_series > val_up)
        rule, value = (rule_low, rule_up), (val_low, val_up)

        index = np.arange(data_series.shape[0])[rule[0] | rule[1]]
        # print("Delete number is: {}".format(len(index)))
        data_m = data_m.drop(index)
        data_m.reset_index(drop=True, inplace=True)
        # print('boundaries:', min(data_m[col_name]), 'to', max(data_m[col_name]))
        return data_m

    @staticmethod
    def remove_space(df, col):
        """
            去除某列中的空格
        """
        newName = df[col].str.strip()
        df[col] = newName
        return df

    @staticmethod
    def remove_str_num(df, col, type_r):
        """
            去除指定字符或数字
        """
        # remove_type_dict = {0: 'keep letter', 1: 'keep num', 2: 'keep letter and num'
        if type_r == 0:
            df[col] = df[col].apply(lambda x: ''.join(filter(str.isalpha, str(x))))

        elif type_r == 1:
            df[col] = df[col].apply(lambda x: ''.join(filter(str.isdigit, str(x))))

        elif type_r == 2:
            df[col] = df[col].apply(lambda x: ''.join(filter(str.isalnum, str(x))))
        return df

    @staticmethod
    def rename_column(df_name, col, new_col):
        """
            重命名列
        """
        df_m = df_name
        df_m.rename(columns={col: new_col}, inplace=True)
        return df_m

    @staticmethod
    def sort_values(df, col):
        """
        数据排序
        :param df:
        :param col:
        :return:
        """
        df = df.sort_values(by=[col])
        return df

    @staticmethod
    def three_sigma(df, col):
        """
            three sigma异常值分析
        """
        ser = df[col]
        bool_id = ((ser.mean() - 3 * ser.std()) <= ser) & (ser <= (ser.mean() + 3 * ser.std()))
        bool_dict = dict(bool_id)

        true_list = [key for key, value in bool_dict.items() if value]
        false_list = [key for key, value in bool_dict.items() if not value]

        df_true = df.iloc[true_list[0]:true_list[-1] + 1]
        if len(false_list) != 0:
            df_false = df.iloc[false_list[0]:false_list[-1] + 1]
        else:
            df_false = pd.DataFrame()

        print(df_true)
        print(df_false)

        return df_true, df_false
