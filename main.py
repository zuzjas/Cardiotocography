import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

# MISSING VALUES PROBLEM SOLUTION
# Fill missing values with column's mean
def fill_missing_values_mean(dataset, X):
    X = X.fillna(value=X.mean())
    return X

# Fill missing values with column's median
def fill_missing_values_median(dataset, X):
    X = X.fillna(value=X.median())
    return X

# Delete missing values
def delete_missing_values(dataset, X):
    X_dropped = X.dropna(axis=0)
    return X_dropped


#TODO
# FEATURE SELECTION
def recursive_feature_selection(dataset):
    # Remove correlated features
    correlated_features = set()
    correlation_matrix = dataset.drop('1', axis=1).corr()

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i,j])>0.8:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)

    dataset.drop(correlated_features, inplace=True, axis=1)

    # Run RFECV
    X = dataset.drop('1', axis=1)
    y = dataset['1']

    rfc = RandomForestClassifier(random_state=101)
    rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='accuracy')
    rfecv.fit(X, y)
    print('Optimal number of features: {}'.format(rfecv.n_features_))

    return dataset

if __name__ == '__main__':
    # Load data with named columns
    dataset = pd.read_csv('ctg_3_col.csv')

    X = dataset.drop(columns="15")
    y = dataset["15"]

    # Solve missing values problem
    X = fill_missing_values_median(dataset,X)
    #X = delete_missing_values(dataset, X)

    # Select features
    X = recursive_feature_selection(X)

    # Save result to csv
    dataset = X
    dataset.to_csv('output.csv')
    print(dataset)

