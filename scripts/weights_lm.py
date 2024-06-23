import pandas as pd
from src.x_reporto.data_loader.tokenizer import Tokenizer
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

dataset_path = 'datasets/train_full.csv'
data_info = pd.read_csv(dataset_path, header=None)
data_info = data_info.iloc[1:]


tokenizer = Tokenizer('healx/gpt-2-pubmed-medium')

# create array of length 50257
vocab_size = 50258

freq = [0] * vocab_size
# loop on boxes phrases
for idx in range(len(data_info)):
    bbox_phrases = data_info.iloc[idx, 6]
    try:
        # convert the string representation of labels into list
        bbox_phrases = eval(bbox_phrases)
        tokenize_phrase = tokenizer(bbox_phrases)
        # tokenize_phrase is a list of lists of integers
        # loop on the lists of integers
        for phrase in tokenize_phrase["input_ids"]:
            for token in phrase:
                if token == 50256:
                    continue
                # print(token)
                freq[token] += 1
        
        # print progress bar
        print(f"Progress: {idx}/{len(data_info)}", end="\r")
    except Exception as e:
        # create a list of empty strings of size 29
        print(e)



# print most frequent words
most_freq = sorted(range(len(freq)), key=lambda i: freq[i], reverse=True)[:100]

for i in most_freq:
    print(tokenizer.tokenizer.decode(i), freq[i])
# calculate the weights
total = sum(freq)

for i in range(len(freq)):
    freq[i] = total / ((freq[i] + 1)* vocab_size)


#plot histogram in range 0 to 50256 values only
plt.hist(freq, bins=100)
plt.show()

# save the weights
np.save('weights.npy', freq)

# load the weights
weights = np.load('weights.npy')
