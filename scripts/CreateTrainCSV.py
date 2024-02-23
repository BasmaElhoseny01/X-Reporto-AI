# Example command to run the script:
# python fix_csv.py --csv "train.csv" --new_csv "train.csv"

import pandas as pd
import argparse
import os
import cv2 as cv
import os
import pandas as pd
from tqdm import tqdm


def check_images_exist(df):
    # Current Working Directory
    current_directory = os.getcwd()
    # Create a boolean Series with all True values initially
    condition = pd.Series(True, index=df.index)

    # Use tqdm to add a progress bar to the loop
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Checking Images"):
        img_file_path = os.path.join(current_directory, row['mimic_image_file_path'])
        
        # Replace backslashes with forward slashes
        img_file_path = img_file_path.replace("\\", "/")
        
        if not os.path.exists(img_file_path):
            condition[index] = False
            print("Removing.. ", row['mimic_image_file_path'])

    print("\n\nold len of csv", len(df))
    df = df[condition]
    print("new len of csv", len(df))

    return df


   

def main(csv_name,new_csv_name):
    # Get the current working directory
    current_directory = os.getcwd()
    print(current_directory)

    # # URL of the CSV file
    csv_path = os.path.join(current_directory,csv_name)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Remove Row with no images
    df=check_images_exist(df)
    # print(df)

    # Fix Bounding Boxes Ratio Problem
    # df=check_bounding_boxes(df)

    # Save new CSV File
    df.to_csv(new_csv_name, index=False)




# /home/basma/Desktop/csv_processing/datasets/mimic-cxr-jpg\files/p10/p10144424/s55318406/0b3d1365-cecec550-e53e23f3-2fd4e293-81245c21.jpg
# /home/basma/Desktop/csv_processing/datasets/mimic-cxr-jpg/files/p10/p10144424/s55318406/0b3d1365-cecec550-e53e23f3-2fd4e293-81245c21.jpg
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleaning CSV file.")
    parser.add_argument("--csv", help="Name of the csv file to cleaned",default='./datasets/train.csv')
    parser.add_argument("--new_csv", help="New name for the CSV file", default='./datasets/train.csv')

    args = parser.parse_args()

    if args.new_csv is None:
        args.new_csv=os.path.splitext(args.csv)[0]+'_clean.csv'

    print("New csv is...",args.new_csv,'\n')

    # Call the main function with the provided file path argument
    main(csv_name=args.csv,new_csv_name=args.new_csv)

# /home/basma/Desktop/csv_processing/datasets/mimic-cxr-jpg/files/p10/p10144424/s52247265/67aaccfc-ca39403f-ed3a91d0-b0467d05-54914729.jpg
                                #    datasets/mimic-cxr-jpg\files/p10/p10144424/s52247265/67aaccfc-ca39403f-ed3a91d0-b0467d05-54914729.jpg