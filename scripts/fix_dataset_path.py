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

New folder hierarchy:
- Graduation-Project
  |- datasets
     |- mimic-cxr-jpg
        |- files
           |- p10
           |- p11

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

    >>>  /home/basma/Desktop/Graduation-Project$ python .\scripts\fix_dataset_path.py
"""

import shutil
import os
import sys
import re

def main():

    # 1. Remove 2.0.0
    parent_folder = "./datasets/physionet.org/files/mimic-cxr-jpg/"
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
    parent_folder = "./datasets/"
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

        # Iterate over the files and remove .txt files
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(parent_folder, 'physionet.org', file)
                print(file_path)
                os.remove(file_path)


        # This isn't extra
        os.rmdir(os.path.join(parent_folder, 'physionet.org'))



if __name__ == "__main__":
    # Get the current working directory
    current_directory = os.getcwd()
    # print(current_directory)

    # Use regex to capture the directory path
    match = re.search(r'.*\\Graduation-Project$', current_directory)
    # print(match)
    
    if not match:
        print("Error: This script should be called from the 'Graduation-Project' directory.")
        sys.exit(1)
    

    print("main func")

    # Call the main function
    # main()