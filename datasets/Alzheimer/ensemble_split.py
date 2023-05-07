import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

# Carica il file CSV
file_path = 'Alzheimer_normalized.csv'
data = pd.read_csv(file_path)

# Dividi il dataset in feature e target
X = data.drop('target', axis=1)
y = data['target']

# Dividi il dataset in Training Set A (70%) e Test Set B (30%) utilizzando uno split stratificato
X_train_A, X_test_B, y_train_A, y_test_B = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
train_set_A = X_train_A.join(y_train_A)
test_set_B = X_test_B.join(y_test_B)

# Crea le cartelle per i folds del Training Set A e del Test Set B
os.makedirs('folds_train_set_A', exist_ok=True)
os.makedirs('folds_test_set_B', exist_ok=True)

# Applica la 5-fold cross-validation stratificata sul Training Set A
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

for fold_number, (train_index, validation_index) in enumerate(skf.split(X_train_A, y_train_A), start=1):
    train_fold = train_set_A.iloc[train_index]
    validation_fold = train_set_A.iloc[validation_index]

    # Salva i folds del Training Set A in file CSV separati
    train_fold.to_csv(f'folds_train_set_A/train_fold_{fold_number}.csv', index=False)
    validation_fold.to_csv(f'folds_train_set_A/validation_fold_{fold_number}.csv', index=False)

    # Esegui i primi n test utilizzando train_fold e validation_fold
    # e seleziona il migliore individuo per ciascuna iterazione
    # ...

# Applica la 5-fold cross-validation stratificata sul Test Set B
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

for fold_number, (train_index, validation_index) in enumerate(skf.split(X_test_B, y_test_B), start=1):
    train_fold = test_set_B.iloc[train_index]
    validation_fold = test_set_B.iloc[validation_index]

    # Salva i folds del Test Set B in file CSV separati
    train_fold.to_csv(f'folds_test_set_B/train_fold_{fold_number}.csv', index=False)
    validation_fold.to_csv(f'folds_test_set_B/validation_fold_{fold_number}.csv', index=False)

    # Esegui l'addestramento e la selezione del migliore individuo
    # utilizzando train_fold e validation_fold
    # ...
