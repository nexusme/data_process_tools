import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import precision_recall_curve, roc_curve, auc


class GenerateCharts(object):
    """
    保存图片模块
    生成图 保存文件
    """

    @staticmethod
    def save_box_plot(df_name, path):
        """
        :param path: 保存路径
        :param df_name: dataframe
        :return:
        """
        plt.rcParams['axes.unicode_minus'] = False  # 显示负号
        pp = PdfPages(path + 'box_plot.pdf')
        cols = df_name.columns[0:]
        print(cols)
        for column in cols:
            plt.figure()
            plt.boxplot(df_name[column], sym='r*')
            plt.grid(True)
            plt.title(column)
            pp.savefig()
        pp.close()

    @staticmethod
    def save_hist(df_name, path):
        """
        画出直方图
        :param path: 保存路径
        :param df_name: dataframe
        :return:
        """
        plt.rcParams['axes.unicode_minus'] = False  # 显示负号
        pp = PdfPages(path + "hist.pdf")
        cols = df_name.columns[0:]
        for column in cols:
            plt.figure()
            plt.hist(df_name[column])
            plt.grid(True)
            plt.title(column)
            pp.savefig()
        pp.close()

    @staticmethod
    def save_heat_map(df_name, path):
        """
        画热力图
        :param path:
        :param df_name: dataframe
        :return:
        """
        # 计算协方差
        # print(df_name.corr())
        # 计算协方差
        plt.rcParams['axes.unicode_minus'] = False  # 显示负号
        pp = PdfPages(path + "heat_map.pdf")
        corr = df_name.corr()
        sns.heatmap(corr,
                    annot=True,
                    annot_kws={'size': 5, 'weight': 'bold', 'color': 'blue'},
                    xticklabels=corr.columns.values,
                    yticklabels=corr.columns.values,
                    cmap='rainbow'
                    )  # 画热力图

        plt.xticks(rotation=90, size=4)  # 将字体进行旋转
        plt.yticks(rotation=360, size=5)
        pp.savefig()
        pp.close()

    @staticmethod
    def save_PR(Y_test, Y_score, path):
        """
        绘制PR图
        :param path:
        :param Y_test: 测试集y
        :param Y_score: 预测得分y
        :return:
        """
        plt.rcParams['axes.unicode_minus'] = False  # 显示负号
        pp = PdfPages(path + "PR.pdf")
        plt.figure(1)  # 创建图表1
        plt.title('Precision/Recall Curve')  # give plot a title
        plt.xlabel('Recall')  # make axis labels
        plt.ylabel('Precision')
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label="Luck")  # 画对角线
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        Y_test = np.array(Y_test)
        precision, recall, thresholds = precision_recall_curve(np.array(Y_test), Y_score)
        plt.figure(1)
        plt.plot(recall, precision)
        pp.savefig()
        pp.close()

    @staticmethod
    def save_ROC_curve(Y_test, Y_score, path):
        """
        绘制ROC曲线
        :param path:
        :param Y_test: 测试集y
        :param Y_score: 得分y
        :return:
        """
        plt.rcParams['axes.unicode_minus'] = False  # 显示负号
        pp = PdfPages(path + "ROC.pdf")
        Y_test = np.array(Y_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_score)
        print('y test:', Y_test)
        print('y_predict', Y_score)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        plt.title('ROC')
        plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        pp.savefig()
        pp.close()

    @staticmethod
    def save_KS_curve(Y_test, Y_score, path):
        """
        绘制KS曲线
        :param path:
        :param Y_test: 测试集y
        :param Y_score: 预测y
        :return:
        """
        fpr, tpr, thresholds = roc_curve(Y_test, Y_score)
        ks_value = max(abs(fpr - tpr))
        plt.rcParams['axes.unicode_minus'] = False  # 显示负号
        pp = PdfPages(path + "KS.pdf")
        # 画图，画出曲线
        plt.plot(fpr, label='bad')
        plt.plot(tpr, label='good')
        plt.plot(abs(fpr - tpr), label='diff')
        # 标记ks
        x = np.argwhere(abs(fpr - tpr) == ks_value)[0, 0]
        plt.plot((x, x), (0, ks_value), label='ks - {:.2f}'.format(ks_value), color='r', marker='o',
                 markerfacecolor='r',
                 markersize=5)
        plt.scatter((x, x), (0, ks_value), color='r')
        plt.legend()
        pp.savefig()
        pp.close()
        print('ks:', ks_value)
        return ks_value
