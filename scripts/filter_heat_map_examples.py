import pandas as pd
import os
import numpy as np

import argparse
"""
Description:
------------
This script filters a CSV file based on indices provided in a separate .npy file and saves the resulting filtered CSV to a new file.

Usage:
------
To run this script, execute the following command from the terminal:

    python ./scripts/filter_heat_map_examples.py --csv <name_of_csv_file> --index <name_of_indices_file> --new_csv <name_for_new_csv_file>

Example:
--------
To run the script, execute:

    >>> python ./scripts/filter_heat_map_examples.py --csv ./datasets/data_server/heat_map_val_balanced.csv --index ./datasets/best_examples.npy --new_csv ./datasets/heat_map_index_filtered.csv

Directory:
----------
This script should be located in the `/Graduation-Project` directory within the project.
"""

def main(csv_name,indices_file_name,new_csv_name):
    # Get the current working directory
    current_directory = os.getcwd()
    print(current_directory)

    # # URL of the CSV file
    csv_path = os.path.join(current_directory,csv_name)
    print("Reading CSV from: ", csv_path)
    # Read Csv 
    df = pd.read_csv(csv_path)

    # Read Indices
    # # URL of the np file
    np_path = os.path.join(current_directory,indices_file_name)
    print("Reading Indices from: ", np_path)
    indices = np.load(np_path)



    # Filter the CSV
    new_df = df.iloc[indices]

    # Save new CSV File
    new_df.to_csv(new_csv_name, index=False)
    print(f"New CSV File saved as {new_csv_name}")


    print(f"Number of rows in the old csv: {len(df)}")
    print(f"Number of rows in the new csv: {len(new_df)}")

    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter HeatMap CSV based on indices File.")
    parser.add_argument("--csv", help="Name of the csv file to cleaned",default='./datasets/heat_map_val.csv')
    parser.add_argument("--index", help="indices file", default='./datasets/best_examples.npy')
    parser.add_argument("--new_csv", help="New name for the CSV file", default='./datasets/heat_map_index_filtered.csv')

    main(parser.parse_args().csv,parser.parse_args().index,parser.parse_args().new_csv)


 

# PS D:\Graduation-Project> python ./scripts/filter_heat_map_examples.py --csv ./datasets/data_server/heat_map_val.csv --index ./datasets/best_examples.npy --new_csv ./datasets/heat_map_val_best.csv