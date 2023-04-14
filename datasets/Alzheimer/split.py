import pandas as pd
from sklearn.model_selection import train_test_split

# Read the CSV file
file_name = 'Alzheimer_normalized.csv'
df = pd.read_csv(file_name)

# Define the target column (change this to match your target column name)
target_column = 'target'

# Split the data into train and test sets in a stratified way
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_column])

# Save the train and test sets as separate CSV files
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
