# this file should be in the datasets folder
import pandas as pd
import os
path='datasets\HeatMapData.csv'
# read csv file
df = pd.read_csv(path)

# get all values of mimic_image_file_path column
mimic_image_file_path = df['mimic_image_file_path']
# loop through all values of mimic_image_file_path
for value in mimic_image_file_path:
    # check if the path is valid or not
    if os.path.exists(value):
        continue
    else:
        # remove the row if the path is not valid
        df = df[df['mimic_image_file_path'] != value]

# save the new dataframe to a new csv file
df.to_csv(path, index=False)

