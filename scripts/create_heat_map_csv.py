import pandas as pd
import argparse

def merge_csv(df1, df2):
    # Merge the DataFrames
    mergedColumn = ["subject_id","study_id"]
    merged_df = pd.merge(df1, df2, on=mergedColumn)

    # Select the columns to keep
    merged_df = merged_df[['subject_id', 'study_id', 'Atelectasis', 'Cardiomegaly',
       'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
       'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
       'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices',
       'image_id', 'mimic_image_file_path', ]]
    

    return merged_df


# Script to merge chexpert with the data csv to produce labels
if __name__ == '__main__':
    # Take csv path as argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, help='Path to the csv file')
    parser.add_argument('--new_csv_path', type=str, help='Path to the new csv file')

    args = parser.parse_args()

    # Read the CSV files
    df1 = pd.read_csv('datasets/mimic-cxr-2.0.0-chexpert.csv')
    df2 = pd.read_csv(args.csv_path)
    merged_csv = merge_csv(df1, df2)


    # Print len of the original csv files and the merged csv file
    print("Length of Original Csv", len(df2))
    print("Length of Merged Csv", len(merged_csv))


    # Write the merged DataFrame to a new CSV file
    merged_csv.to_csv(args.new_csv_path, index=False)

    print("CSV files merged successfully!") 


#  python .\scripts\create_heat_map_csv.py --csv_path "./datasets/train.csv" --new_csv_path "./datasets/heat_map_train.csv"
