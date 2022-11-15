import pandas as pd
if __name__ == '__main__':
    # Load data with named columns
    dataset = pd.read_csv('ctg_3_col.csv')

    # Fill missing values with column's mean
    X = dataset.drop(columns="15")
    y = dataset["1"]
    X = X.fillna(value=X.mean())

    # Save result to csv
    dataset = X
    dataset.to_csv('output.csv')
    print(dataset)

