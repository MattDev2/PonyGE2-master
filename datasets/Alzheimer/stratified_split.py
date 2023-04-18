import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Load the CSV file as a DataFrame
file_path = './Alzheimer_normalized.csv'
data = pd.read_csv(file_path)

# Identify the feature columns and the target column
X = data.iloc[:, :-1]  # Feature columns
y = data.iloc[:, -1]   # Target column

# Perform the 5-fold stratified split using the StratifiedKFold class
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold = 1
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    train_data.to_csv(f"./folds/train_data_fold_{fold}.csv", index=False)
    test_data.to_csv(f"./folds/test_data_fold_{fold}.csv", index=False)

    fold += 1

    # Train and evaluate your model on the (X_train, y_train) and (X_test, y_test) data
    # Example: model.fit(X_train, y_train) and model.evaluate(X_test, y_test)
    # Repeat this process for all 5 folds
