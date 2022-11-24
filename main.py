import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def make_experiment2(dataset):
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    clfs = {
        'GNB': GaussianNB(),
        'rfc': RandomForestClassifier(random_state=101),
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
            X_train = fill_missing_values_mean(X[train])
            y_train = y[train]
            X_test = fill_missing_values_mean(X[test])
            y_test = y[test]

            # FEATURE SELECTION
            # Recursive feature elimination
            # rfe = RFE(estimator=RandomForestClassifier(random_state=101), n_features_to_select=25)
            # rfe.fit(X_train, y_train)
            # X_train = rfe.transform(X_train)
            # rfe.fit(X_test, y_test)
            # X_test = rfe.transform(X_test)

            # kBest selection
            select = SelectKBest(score_func=f_classif, k=25)
            X_train = select.fit_transform(X_train, y_train)
            X_test = select.fit_transform(X_test, y_test)

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            scores[clf_id, fold_id] = accuracy_score(y_test, y_pred)

    mean = np.mean(scores, axis=1)
    std = np.std(scores, axis=1)

    for clf_id, clf_name in enumerate(clfs):
        print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))


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
