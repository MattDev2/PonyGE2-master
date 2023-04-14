import pandas as pd

# Read the CSV file
file_name = 'Alzheimer_normalized.csv'
df = pd.read_csv(file_name)

# Move the column with header 'target' to the end
column_name = 'target'
if column_name in df.columns:
    df[column_name] = df.pop(column_name)

# Save the modified DataFrame to the same input CSV file
df.to_csv(file_name, index=False)
