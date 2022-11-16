import pandas as pd


# Fill missing values with column's mean
def fill_missing_values(dataset, X):
    X = X.fillna(value=X.mean())
    return X


# Delete missing values
def delete_missing_values(dataset, X):
    X_dropped = X.dropna(axis=0)
    return X_dropped


if __name__ == '__main__':
    # Load data with named columns
    dataset = pd.read_csv('ctg_3_col.csv')

    X = dataset.drop(columns="15")
    y = dataset["15"]

    # Solve missing values problem
    X = fill_missing_values(dataset,X)
    #X = delete_missing_values(dataset, X)

    # Save result to csv
    dataset = X
    dataset.to_csv('output.csv')
    print(dataset)

