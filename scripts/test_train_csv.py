# load train.csv and test.csv and check if the data is correct
import pandas as pd
from tqdm import tqdm

train = pd.read_csv('./datasets/train.csv')

print(train.head())
# new empty csv

new_train = pd.DataFrame()

# add headers to the new csv file
new_train = new_train.append(train.iloc[0])
print("header: ",train.iloc[0])
for i in tqdm(range(1,len(train))):
    # check if box column is not empty
    try:
        if train.iloc[i, 4] != ' ':
            boxes = eval(train.iloc[i, 4])
            # check if boxes have shape (N,4)

            if len(boxes) > 0:
                if len(boxes[0]) == 4:
                    #add the row to the new csv file
                    new_train = new_train.append(train.iloc[i])
                else:
                    print("error at: ", i)
                    print(boxes)
                    continue
            else:
                print("error at: ", i)
                print(boxes)
                continue
        else:
            print("error at: ", i)
            print(train.iloc[i, 4])
            continue
    except Exception as e:
        print(e)
        continue 
# save the new file
new_train.to_csv('./datasets/new_train', index=False)
print(new_train.head())