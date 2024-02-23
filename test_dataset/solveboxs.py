import pandas as pd
import sys
from tqdm import tqdm 
# read 2 csv files
file1= 'test_dataset/train.csv'
file2= 'test_dataset/box_train.csv'
new_file = 'test_dataset/newtest.csv'
# read the csv files
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
# new empty csv
new_df = pd.DataFrame()
# index for bboxs column
box=4

# loop through the rows of the first csv file if the first 2 columns are equal with the first 2 columns of the second csv file
# then replace third column in the df1 with the 3 column in the df2
for i in tqdm(range(len(df1))):
    if df1.iloc[i,0] == df2.iloc[i,0] and df1.iloc[i,1] == df2.iloc[i,1]:
        # insert the row in the 1 csv file to the 3 csv file
        new_df = new_df.append(df1.iloc[i])
        # replace the 3 column in the new csv file with the 3 column in the 2 csv file
        new_df.iloc[i, box] = df2.iloc[i, box]
    else :
        print(df1.iloc[i,0],"   ", df2.iloc[i,0],"   ",df1.iloc[i,1],"   ", df2.iloc[i,1])
# save the new file
new_df.to_csv(new_file, index=False)

