import pandas as pd

# Read the CSV files
df1 = pd.read_csv('datasets/mimic-cxr-2.0.0-chexpert.csv')
df2 = pd.read_csv('datasets/train_full.csv')

# Merge the DataFrames
mergedColumn = ["subject_id","study_id"]
merged_df = pd.merge(df1, df2, on=mergedColumn)

# remove column number 10 from csv file
merged_df=merged_df.drop(merged_df.columns[10],axis=1)

# Write the merged DataFrame to a new CSV file
merged_df.to_csv('datasets/heat_map_full.csv', index=False)