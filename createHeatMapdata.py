import pandas as pd

# # Read the CSV files
# df1 = pd.read_csv('datasets/mimic-cxr-2.0.0-chexpert.csv/mimic-cxr-2.0.0-chexpert.csv')
# df2 = pd.read_csv('datasets/train.csv')

# # Merge the DataFrames
# meargColumn = ["subject_id","study_id"]
# merged_df = pd.merge(df1, df2, on=meargColumn)

# # Write the merged DataFrame to a new CSV file
# merged_df.to_csv('datasets\HeatMapData.csv', index=False)

# read csv file 
df = pd.read_csv('datasets\HeatMapData.csv')
# remove colomn number 10 from csv file
df = df.drop(df.columns[10], axis=1)
# save the new csv file
df.to_csv('datasets\HeatMapData.csv', index=False)