import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from nltk.tokenize import RegexpTokenizer
import re

# Function to load a CSV file and return the DataFrame excluding the header row
def load_csv(path):
    df = pd.read_csv(path, header=None)
    print("number of examples: ",len(df)-1)
    return df.iloc[1:]

# Function to get the general statistics from the DataFrame
def get_general_stats(data_frame):
    # loop over the dataframe
    dataset_size = len(data_frame)
    count_abnormal = 0
    count_normal = 0
    count_bbox_phrase_exists = 0
    count_bbox_phrase_not_exists = 0

    # Arrays to hold counts for different conditions
    abnormal_array = [0]*29
    normal_array = [0]*29
    bbox_phrase_exists_array = [0]*29
    bbox_phrase_not_exists_array = [0]*29

    abnormal_phrase_exists_array = [0]*29
    abnormal_phrase_not_exists_array = [0]*29
    normal_phrase_exists_array = [0]*29
    normal_phrase_not_exists_array = [0]*29

    # Loop over the DataFrame to accumulate statistics
    for i in range(dataset_size):
        bboxes = data_frame.iloc[i, 4]
        # convert the string representation of bounding boxes into list of list
        bboxes = eval(bboxes)

        # get labels
        labels = data_frame.iloc[i, 5]
        # convert the string representation of labels into list
        labels = np.array(eval(labels))

        # get bbox is abnormal
        bbox_is_abnormal = data_frame.iloc[i, 8]

        # convert the string representation of bbox_is_abnormal into list
        bbox_is_abnormal = eval(bbox_is_abnormal)    

        # get bbox_phrase_exists
        bbox_phrase_exists = data_frame.iloc[i, 7]

        # convert the string representation of bbox_phrase_exists into list
        bbox_phrase_exists = eval(bbox_phrase_exists)
        
        count_abnormal += sum(bbox_is_abnormal)
        count_normal += len(bbox_is_abnormal) - sum(bbox_is_abnormal)

        count_bbox_phrase_exists += sum(bbox_phrase_exists)
        count_bbox_phrase_not_exists += len(bbox_phrase_exists) - sum(bbox_phrase_exists)
        
        # Update arrays based on conditions
        for i in range(len(bbox_is_abnormal)):
            if bbox_is_abnormal[i] == 1:
                abnormal_array[i] += 1
            else:
                normal_array[i] += 1
        for i in range(len(bbox_phrase_exists)):
            if bbox_phrase_exists[i] == 1:
                bbox_phrase_exists_array[i] += 1
            else:
                bbox_phrase_not_exists_array[i] += 1
        for i in range(len(bbox_phrase_exists)):
            if bbox_is_abnormal[i] == 1:
                if bbox_phrase_exists[i] == 1:
                    abnormal_phrase_exists_array[i] += 1
                else:
                    abnormal_phrase_not_exists_array[i] += 1
            else:
                if bbox_phrase_exists[i] == 1:
                    normal_phrase_exists_array[i] += 1
                else:
                    normal_phrase_not_exists_array[i] += 1
    
    print("abnormal_array: ",abnormal_array)
    print("normal_array: ",normal_array)

    # normalized array
    abnormal_array = np.array(abnormal_array)
    normal_array = np.array(normal_array)
    abnormal_array = abnormal_array/(abnormal_array+normal_array)
    print("abnormal_array_normalized: ",abnormal_array)
    print("normal_array_normalized: ",1-abnormal_array)

    print("bbox_phrase_exists_array: ",bbox_phrase_exists_array)
    print("bbox_phrase_not_exists_array: ",bbox_phrase_not_exists_array)

    # print normalized array
    bbox_phrase_exists_array = np.array(bbox_phrase_exists_array)
    bbox_phrase_not_exists_array = np.array(bbox_phrase_not_exists_array)
    bbox_phrase_exists_array = bbox_phrase_exists_array/(bbox_phrase_exists_array+bbox_phrase_not_exists_array)
    print("bbox_phrase_exists_array_normalized: ",bbox_phrase_exists_array)
    print("bbox_phrase_not_exists_array_normalized: ",1-bbox_phrase_exists_array)
    

    print("abnormal_phrase_exists_array: ",abnormal_phrase_exists_array)
    print("abnormal_phrase_not_exists_array: ",abnormal_phrase_not_exists_array)
    print("normal_phrase_exists_array: ",normal_phrase_exists_array)
    print("normal_phrase_not_exists_array: ",normal_phrase_not_exists_array)

    # print normalized array
    abnormal_phrase_exists_array = np.array(abnormal_phrase_exists_array)
    abnormal_phrase_not_exists_array = np.array(abnormal_phrase_not_exists_array)
    normal_phrase_exists_array = np.array(normal_phrase_exists_array)
    normal_phrase_not_exists_array = np.array(normal_phrase_not_exists_array)
    abnormal_phrase_exists_array = abnormal_phrase_exists_array/(abnormal_phrase_exists_array+abnormal_phrase_not_exists_array)
    abnormal_phrase_not_exists_array = abnormal_phrase_not_exists_array/(abnormal_phrase_exists_array+abnormal_phrase_not_exists_array)
    normal_phrase_exists_array = normal_phrase_exists_array/(normal_phrase_exists_array+normal_phrase_not_exists_array)
    normal_phrase_not_exists_array = normal_phrase_not_exists_array/(normal_phrase_exists_array+normal_phrase_not_exists_array)
    print("abnormal_phrase_exists_array_normalized: ",abnormal_phrase_exists_array)
    print("abnormal_phrase_not_exists_array_normalized: ",abnormal_phrase_not_exists_array)
    print("normal_phrase_exists_array_normalized: ",normal_phrase_exists_array)
    print("normal_phrase_not_exists_array_normalized: ",normal_phrase_not_exists_array)

    print("count_abnormal: ",count_abnormal)
    print("count_normal: ",count_normal)
    print("count_bbox_phrase_exists: ",count_bbox_phrase_exists)
    print("count_bbox_phrase_not_exists: ",count_bbox_phrase_not_exists)

    pos_weight = count_normal/count_abnormal
    pos_phrase_weight = count_bbox_phrase_not_exists/count_bbox_phrase_exists
    return count_abnormal/(count_abnormal+count_normal), count_bbox_phrase_exists/(count_bbox_phrase_exists+count_bbox_phrase_not_exists), pos_weight,pos_phrase_weight

# Function to get the bounding box statistics from the DataFrame
def get_bboxes_stats(data_frame):
    # loop over the dataframe
    dataset_size = len(data_frame)
    bboxes_histogram_width = { i: [0]*224 for i in range(29)}
    bboxes_histogram_height = { i: [0]*224 for i in range(29)}
    for i in range(dataset_size):
        bboxes = data_frame.iloc[i, 4]
        # convert the string representation of bounding boxes into list of list
        bboxes = eval(bboxes)

        # loop over the bboxes
        for label,bbox in enumerate(bboxes):
            # get the width and height of the bbox
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            # update the histogram
            bboxes_histogram_width[label][width] += 1
            bboxes_histogram_height[label][height] += 1
    # for each label plot the histogram
    for label in range(24,29):
        # plot the width histogram
        plt.bar(range(224),bboxes_histogram_width[label])
        plt.title(f"width histogram for label {label}")
        plt.show()
        # plot the height histogram
        plt.bar(range(224),bboxes_histogram_height[label])
        plt.title(f"height histogram for label {label}")
        plt.show()
    return bboxes_histogram_width,bboxes_histogram_height

def get_phrase_stats(data_frame):
    # loop over the dataframe
    dataset_size = len(data_frame)
    max_phrase_length = 0
    min_phrase_length = 1024
    phrase_histogram = [0]*1024
    for i in range(dataset_size):
        image_phrases = data_frame.iloc[i, 6]
        # convert the string representation of bounding boxes into list of list
        image_phrases = eval(image_phrases)
        for phrase in image_phrases :
            if phrase=="":
                continue
            max_phrase_length = max(max_phrase_length,len(phrase.split(" ")))
            min_phrase_length = min(min_phrase_length,len(phrase.split(" ")))
            phrase_histogram[len(phrase.split(" "))] += 1
    print("histogram: ",phrase_histogram)
    plt.bar(range(1024),phrase_histogram)
    plt.title(f"phrase histogram")
    plt.show()
    return max_phrase_length,min_phrase_length



def get_phrase_stats_boxes_histogram(data_frame):
    # loop over the dataframe
    dataset_size = len(data_frame)
    phrase_histogram = [0]*30
    for i in range(dataset_size):
        image_phrases = data_frame.iloc[i, 6]
        # convert the string representation of bounding boxes into list of list
        image_phrases = eval(image_phrases)
        count=0
        for j,phrase in enumerate(image_phrases) :
            if phrase!="":
                count += 1
        phrase_histogram[count] += 1
    print("histogram: ",phrase_histogram)
    plt.bar(range(30),phrase_histogram)
    plt.title(f"phrase histogram")
    plt.show()
    return phrase_histogram

if __name__ == '__main__':
    # load the csv file
    csv_path = os.path.join(os.getcwd(), "datasets/train.csv")
    df = load_csv(csv_path)
    print(get_phrase_stats(df))
    # get the stats
    abnormal_ratio,phrase_exists_ratio,pos_weight,pos_phrase_weight = get_general_stats(df)

    print("abnormal_ratio: ",abnormal_ratio)
    print("phrase_exists_ration: ",phrase_exists_ratio)
    print("pos_weight: ",pos_weight)
    print("pos_phrase_weight: ",pos_phrase_weight)

    # get the bboxes stats
    bboxes_histogram_width,bboxes_histogram_height = get_bboxes_stats(df)
    