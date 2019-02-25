import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV

pd.set_option('display.expand_frame_repr', False)


def load_data(dataPath):
    svm_data = pd.read_csv(dataPath, sep=",", header=None)
    svm_data[0] = svm_data[0].replace([0], -1)
    return svm_data


def svm_model(dataPath, parameters, cv):
    svm_data = load_data(dataPath)
    data = svm_data.loc[:, svm_data.columns != 34]
    labels = svm_data.iloc[:, -1]
    svc = svm.SVC()
    clf = GridSearchCV(estimator=svc, param_grid=parameters,
                       cv=cv, return_train_score=True)
    clf.fit(data, labels)
    print(sorted(clf.cv_results_.keys()))
    cv_results = pd.DataFrame.from_dict(clf.cv_results_)
    cv_results.to_csv('output1.csv')

    return cv_results


if __name__ == '__main__':
    dataPath = str(sys.argv[1])
    # dataPath = "F:/University of Waterloo/Winter 2019/CS 680/Assignments/assignment_2/LinearRegression/DataSets/SVM_data/ionosphere.csv"
    parameters = {
                    'kernel': ('linear', 'rbf', 'poly'),
                    'C': [1, 10, 100],
                    'degree': [3,4,5]
                    }
    cv = 10
    gamma_1 = 1
    svm_model(dataPath, parameters, cv)
