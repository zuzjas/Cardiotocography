import numpy as np
import pandas as pd
from pip._internal.utils.misc import tabulate
from scipy.stats import ttest_rel
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate


def make_experiment2(dataset):
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    clfs = {
        'GNB': GaussianNB(),
        'kNN': KNeighborsClassifier(),
        'CART': DecisionTreeClassifier(random_state=42),
    }

    n_splits = 3
    n_repeats = 2
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    scores = np.zeros((len(clfs), n_splits * n_repeats))

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])

            # MISSING VALUES PROBLEM SOLUTION
            X_train = fill_missing_values_median(X[train])
            y_train = y[train]
            X_test = fill_missing_values_median(X[test])
            y_test = y[test]

            # FEATURE SELECTION
            # Recursive feature elimination
            rfe = RFE(estimator=RandomForestClassifier(random_state=101), n_features_to_select=20)
            rfe.fit(X_train, y_train)
            X_train = rfe.transform(X_train)
            rfe.fit(X_test, y_test)
            X_test = rfe.transform(X_test)

            # # kBest selection
            # select = SelectKBest(score_func=f_classif, k=25)
            # X_train = select.fit_transform(X_train, y_train)
            # X_test = select.fit_transform(X_test, y_test)

            # Klasyfikacja
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            scores[clf_id, fold_id] = accuracy_score(y_test, y_pred)

    mean = np.mean(scores, axis=1)
    std = np.std(scores, axis=1)

    for clf_id, clf_name in enumerate(clfs):
        print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))

    np.save("results", scores)

    scores = np.load('results.npy')
    print("Folds:\n", scores)

    # t-statystyka oraz p-wartosc
    alfa = .05
    t_statistic = np.zeros((len(clfs), len(clfs)))
    p_value = np.zeros((len(clfs), len(clfs)))

    for i in range(len(clfs)):
        for j in range(len(clfs)):
            t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])
    print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

    headers = ["GNB", "kNN", "CART"]
    names_column = np.array([["GNB"], ["kNN"], ["CART"]])
    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

    # Przewaga
    advantage = np.zeros((len(clfs), len(clfs)))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    print("Advantage:\n", advantage_table)

    # Różnice statystycznie znaczące
    significance = np.zeros((len(clfs), len(clfs)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    print("Statistical significance (alpha = 0.05):\n", significance_table)

    # Wynik końcowy analizy statystycznej
    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), headers)
    print("Statistically significantly better:\n", stat_better_table)


# Fill missing values with column's mean
def fill_missing_values_mean(dataset):
    col = []
    for i in range(1, len(dataset[0]) + 1):
        col.append(i)
    dataset = pd.DataFrame(data=dataset, columns=col)

    X = dataset.fillna(value=dataset.mean())

    return X


# Fill missing values with column's median
def fill_missing_values_median(dataset):
    col = []
    for i in range(1, len(dataset[0]) + 1):
        col.append(i)
    dataset = pd.DataFrame(data=dataset, columns=col)

    dataset = dataset.fillna(value=dataset.median())
    return dataset


# Delete missing values
def delete_missing_values(dataset):
    col = []
    for i in range(1, len(dataset[0]) + 1):
        col.append(i)
    dataset = pd.DataFrame(data=dataset, columns=col)

    X_dropped = dataset.dropna(axis=0)
    return X_dropped


# def select_from_model_selection(dataset):
#     X = dataset.drop('36', axis=1)
#     y = dataset['36']
#
#     # Run SFM
#     sfm_selector = SelectFromModel(estimator=LogisticRegression())
#     sfm_selector.fit(X, y)
#     print("Selected features: ")
#     print(X.columns[sfm_selector.get_support()])
#
#     # Drop not selected features
#     X.drop(X.columns[np.where(sfm_selector.get_support() == False)[0]], axis=1, inplace=True)
#
#     return X


if __name__ == '__main__':
    # Load data
    dataset = 'ctg_3'
    dataset = np.genfromtxt("%s.csv" % (dataset), delimiter=",")

    # Experiment
    make_experiment2(dataset)
