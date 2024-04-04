"""
Description:
------------
This script cleans the PhysioNet dataset by moving files and removing unnecessary directories and files.

Folder Structure Changes:
-------------------------
Old folder hierarchy:
- Graduation-Project
  |- datasets
     |- physionet.org
        |- files
           |- mimic-cxr-jpg
              |- 2.0.0
                 |- files
                    |- p10
                    |- p11
        |- robots.txt
  |- p10_subset1.zip
  |- p11_subset2.zip

New folder hierarchy:
- Graduation-Project
  |- datasets
     |- mimic-cxr-jpg
        |- files
           |- p10
           |- p11
     |- p10_subset1.zip
     |- p11_subset2.zip

Usage:
------
To run this script, execute the following command from the terminal:

    # python .\scripts\fix_dataset_path.py

Directory:
----------
This script should be located in the `/Graduation-Project` directory within the project.

Example:
--------
To run the script , execute:

    >>>  /home/basma/Desktop/Graduation-Project$ python ./scripts/fix_dataset_path.py
"""

import shutil
import os
import argparse

import sys
import re

def main(parent_folder="./datasets/physionet.org/files/mimic-cxr-jpg/", datasets_parent="./datasets/"):


    # 1. Remove 2.0.0
    parent_folder = parent_folder
    child_folder = os.path.join(parent_folder, '2.0.0/')


    # List files in the child folder
    files = os.listdir(child_folder)
    print(files)

    # Move files to the parent folder
    for file in files:
        source = os.path.join(child_folder, file)
        destination = os.path.join(parent_folder, file)
        print("destination",destination)
        print("source",source)
        shutil.move(source, destination)

        # Remove the now empty child folder
        os.rmdir(child_folder)

    # 2. Remove physionet.org/files
    parent_folder = datasets_parent
    child_folder = os.path.join(parent_folder, 'physionet.org/files/')

        # List files in the child folder
    files = os.listdir(child_folder)
    print(files)

    # Move files to the parent folder
    for file in files:
        source = os.path.join(child_folder, file)
        destination = os.path.join(parent_folder, file)
        print("destination",destination)
        print("source",source)
        shutil.move(source, destination)

        # Remove the now empty child folder
        os.rmdir(child_folder)

        # List all files in the folder
        files = os.listdir(os.path.join(parent_folder, 'physionet.org'))

        # Remove robots.txt file 
        file='robots.txt'
        file_path = os.path.join(parent_folder, 'physionet.org', file)
        print(file_path)
        os.remove(file_path)

        # This isn't extra
        os.rmdir(os.path.join(parent_folder, 'physionet.org'))



if __name__ == "__main__":
    # Get the current working directory
    # current_directory = os.getcwd()
    # print(current_directory)

    # # Use regex to capture the directory path
    # match = re.search(r'.*(\\|/)Graduation-Project$', current_directory)
    # print(match)
    
    # if not match:
    #     print("Error: This script should be called from the 'Graduation-Project' directory.")
    #     sys.exit(1)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--parent_folder', default="./datasets/physionet.org/files/mimic-cxr-jpg/", type=str,
                        help='Parent folder for processing')
    parser.add_argument('--datasets_parent', default="./datasets/", type=str,
                        help='Parent folder for datasets')
    args = parser.parse_args()
    
    
    # Call the main function
    main(args.parent_folder, args.datasets_parent)