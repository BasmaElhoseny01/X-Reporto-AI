# load train.csv and test.csv and check if the data is correct
import pandas as pd
from tqdm import tqdm


train = pd.read_csv('./datasets/train.csv')
train = train.iloc[1:]

# add headers to the new csv file
print("header: ",train.iloc[-1])
# print(train.tail())
