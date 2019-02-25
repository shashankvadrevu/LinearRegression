import sys
import os
import numpy as np
import pandas as pd
from numpy.linalg import inv
import matplotlib.pyplot as plt


def lamda_list(start, stop, step=1):
    n = int(round((stop - start) / float(step)))
    # print(n)
    if n > 1:
        return([round(((start + step * i)), 1) for i in range(n + 1)])
    elif n == 1:
        return([start])
    else:
        return([])


def dataFiles(dataPath):
    x_files = []
    y_files = []
    for root, dirs, files in os.walk(dataPath):
        for eachFile in files:
            if 'fData' in eachFile:
                x_files.append(dataPath + "/" + eachFile)
            elif 'fLabels' in eachFile:
                y_files.append(dataPath + "/" + eachFile)
            else:
                pass
    x_y = []
    for each_xFile, each_yFile in zip(x_files, y_files):
        x_y_each = []
        x_y_each.extend((each_xFile, each_yFile))
        x_y.append(x_y_each)

    return x_y


def leastSquares(dataPath, lamda_start, lambda_stop, lamda_step):
    lamda_values = lamda_list(lamda_start, lambda_stop, lamda_step)
    qualityMeasures_eachLamda = {}
    for eachLamda in lamda_values:
        x_y = dataFiles(dataPath)
        qualityMeasures_eachCV = {}
        for i, each in enumerate(x_y):
            test_file = x_y[i]
            train_file = [x for j, x in enumerate(x_y) if j != i]
            testX_df = pd.read_csv(test_file[0], header=None)
            testY_df = pd.read_csv(test_file[1], header=None)
        # testXY_df = pd.concat([testX_df, testY_df],
        #                       axis=1, join_axes=[testX_df.index])
            trainX_df = pd.DataFrame()
            trainY_df = pd.DataFrame()
            for each in train_file:
                trainX_df_1 = pd.read_csv(each[0], header=None)
                trainY_df_1 = pd.read_csv(each[1], header=None)
                # trainXY_df_1 = pd.concat([trainX_df_1, trainY_df_1],
                #                          axis=1, join_axes=[trainX_df_1.index])
                # trainXY_df = trainXY_df.append(trainXY_df_1, ignore_index=True)
                trainX_df = trainX_df.append(trainX_df_1, ignore_index=True)
                trainY_df = trainY_df.append(trainY_df_1, ignore_index=True)

            X_train = trainX_df.values
            Y_train = trainY_df.values
            X_test = testX_df.values
            Y_test = testY_df.values
            X_train_ones = np.ones(X_train.shape[0], int, 1).reshape(
                X_train.shape[0], 1)
            X_train = np.hstack((X_train, X_train_ones))
            X_test_ones = np.ones(X_test.shape[0], int, 1).reshape(
                X_test.shape[0], 1)
            X_test = np.hstack((X_test, X_test_ones))
            iden = np.identity((X_train.transpose().dot(X_train)).shape[0])
            # calculate coefficients using closed-form solution
            coeffs = inv((X_train.transpose().dot(X_train)) +
                         (eachLamda * iden)).dot(X_train.transpose()).dot(Y_train)
            y_hat_mat = (np.dot(X_test, coeffs))
            # error_eachCV = ((Y_test - y_hat_mat)**2) / 2
            sse = np.sum(((Y_test - y_hat_mat)**2) / 2)
            sst = np.sum(((Y_test - np.mean(Y_test))**2) / 2)
            ssr = np.sum(((y_hat_mat - np.mean(Y_test))**2) / 2)
            n = X_test.shape[0]
            k = X_test.shape[1] - 1
            dof_Error = n - (k + 1)
            unbiased_S_sqrd = sse / dof_Error
            rsqrd_adj = (1 - ((sse / dof_Error) / (sst / n - 1))) * 100

            qualityMeasures_eachCV.setdefault(
                "unbiased_S_sqrd", []).append(unbiased_S_sqrd)
            qualityMeasures_eachCV.setdefault(
                "rsqrd_adj", []).append(rsqrd_adj)
            qualityMeasures_eachCV.setdefault(
                "sse_avg_cv", []).append(sse)

        # print(qualityMeasures_eachCV)
        S_sqrd_avg = round(((sum(
            qualityMeasures_eachCV["unbiased_S_sqrd"]) / len(qualityMeasures_eachCV['unbiased_S_sqrd']))), 5)

        # print(S_sqrd_avg)

        rsqrd_adj_avg = round(((sum(
            qualityMeasures_eachCV["rsqrd_adj"]) / len(qualityMeasures_eachCV['rsqrd_adj']))), 5)

        sse_avg = round(((sum(
            qualityMeasures_eachCV["sse_avg_cv"]) / len(qualityMeasures_eachCV['sse_avg_cv']))), 5)

        # print(rsqrd_adj_avg)
        qualityMeasures_eachLamda.setdefault(
            "unbiased_S_sqrd_avg", []).append(S_sqrd_avg)
        qualityMeasures_eachLamda.setdefault(
            "rsqrd_adj_avg", []).append(rsqrd_adj_avg)
        qualityMeasures_eachLamda.setdefault(
            "sse_avg", []).append(sse_avg)

    return qualityMeasures_eachLamda


def plot_lr(dataPath, lamda_start, lambda_stop, lamda_step):
    qualityMeasures_eachLamda = leastSquares(
        dataPath, lamda_start, lambda_stop, lamda_step)
    lamda_values = lamda_list(lamda_start, lambda_stop, lamda_step)
    cv_plt = plt.figure()
    # plt.plot(lamda_values, qualityMeasures_eachLamda['unbiased_S_sqrd_avg'])
    plt.plot(lamda_values, qualityMeasures_eachLamda['rsqrd_adj_avg'])
    # plt.plot(lamda_values, qualityMeasures_eachLamda['sse_avg'])
    cv_plt.suptitle(
        'Linear Regression Closed Form with 10-Fold CrossVal', fontsize=12)
    cv_plt.legend([
        'rsqrd_adj_avg'], loc='lower left')  # 'sse_avg','unbiased_S_sqrd', 'rsqrd_adj_avg',
    # plt.ylim(85, 102)
    # plt.ylabel('Percent (%)')
    plt.xlabel('Lamda Range')
    plt.show('hold')

    pass


if __name__ == '__main__':
    # dataPath = str(sys.argv[1])
    dataPath = 'F:/University of Waterloo/Winter 2019/CS 680/Assignments/assignment_2/LinearRegression/DataSets/regression-dataset'
    lamda_start = 0
    lambda_stop = 4
    lamda_step = 0.1
    # leastSquares(dataPath, lamda_start, lambda_end, lamda_step)
    plot_lr(dataPath, lamda_start, lambda_stop, lamda_step)
