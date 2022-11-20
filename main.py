import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def make_experiment2(X_sel, dataset):
    dataset = dataset.to_numpy()
    X_sel = X_sel.to_numpy()
    X = X_sel
    y = dataset[:, -1].astype(int)

    clfs = {
        'GNB': GaussianNB(),
        'rfc': RandomForestClassifier(random_state=101),
        'kNN': KNeighborsClassifier(),
        'CART': DecisionTreeClassifier(random_state=42),
    }

    n_splits = 5
    n_repeats = 2
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    scores = np.zeros((len(clfs), n_splits * n_repeats))

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)

    mean = np.mean(scores, axis=1)
    std = np.std(scores, axis=1)

    for clf_id, clf_name in enumerate(clfs):
        print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))


# MISSING VALUES PROBLEM SOLUTION
# Fill missing values with column's mean
def fill_missing_values_mean(dataset):
    X = dataset.fillna(value=dataset.mean())
    return X


# Fill missing values with column's median
def fill_missing_values_median(dataset):
    X = dataset.fillna(value=dataset.median())
    return X


# Delete missing values
def delete_missing_values(dataset):
    X_dropped = dataset.dropna(axis=0)
    return X_dropped


# FEATURE SELECTION
def recursive_feature_selection(dataset):
    print("recursive_feature_selection")
    # Remove correlated features
    correlated_features = set()
    correlation_matrix = dataset.drop('36', axis=1).corr()

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)

    dataset.drop(correlated_features, inplace=True, axis=1)

    # Run RFECV
    X = dataset.drop('36', axis=1)
    y = dataset['36']

    rfc = RandomForestClassifier(random_state=101)
    rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='accuracy')
    rfecv.fit(X, y)
    print('Optimal number of features: {}'.format(rfecv.n_features_))

    # Drop the least important features
    print('The least important features: ')
    print(np.where(rfecv.support_ == False)[0])
    X.drop(X.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)

    # Draw feature importances plot
    dset = pd.DataFrame()
    dset['attr'] = X.columns
    dset['importance'] = rfecv.estimator_.feature_importances_

    dset = dset.sort_values(by='importance', ascending=False)

    plt.figure(figsize=(16, 14))
    plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
    plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Importance', fontsize=14, labelpad=20)
    plt.show()

    return X


def select_from_model_selection(dataset):
    X = dataset.drop('36', axis=1)
    y = dataset['36']

    # Run SFM
    sfm_selector = SelectFromModel(estimator=LogisticRegression())
    sfm_selector.fit(X, y)
    print("Selected features: ")
    print(X.columns[sfm_selector.get_support()])

    # Drop not selected features
    X.drop(X.columns[np.where(sfm_selector.get_support() == False)[0]], axis=1, inplace=True)

    return X


if __name__ == '__main__':
    # Load data with named columns
    dataset = pd.read_csv('ctg_3_col.csv')

    # Solve missing values problem
    dataset = fill_missing_values_median(dataset)
    # dataset = fill_missing_values_mean(dataset)
    # dataset = delete_missing_values(dataset)

    # Select features
    X = recursive_feature_selection(dataset)
    # X = select_from_model_selection(dataset)

    # Save result to csv
    X.to_csv('output.csv')

    # Experiment
    make_experiment2(X, dataset)
