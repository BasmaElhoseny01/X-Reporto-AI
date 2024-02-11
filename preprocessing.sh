#!/bin/bash

echo "Start preprocessing.sh"
echo "Creating folders in paths.py if they don't exist."
python3 - <<END
import os
import shutil
from data_preprocessing.paths import path_chest_imagenome, path_mimic_cxr, path_mimic_cxr_jpg
import subprocess
# create folders in paths if they don't exist
def create_folders(paths):
    print("start")
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print("created successfully path: ",path)

create_folders([path_chest_imagenome, path_mimic_cxr, path_mimic_cxr_jpg])
END

echo "Downloading and unzipping the MIMIC-CXR-JPG dataset."
unzip -o "/content/drive/MyDrive/MIMIC/p11_subset1.zip" -d "datasets/mimic-cxr-jpg"
unzip -o "/content/drive/MyDrive/MIMIC/p11_subset2.zip" -d "datasets/mimic-cxr-jpg"
unzip -o "/content/drive/MyDrive/MIMIC/p11_subset3.zip" -d "datasets/mimic-cxr-jpg"
unzip -o "/content/drive/MyDrive/MIMIC/p11_subset4.zip" -d "datasets/mimic-cxr-jpg"

echo "Renaming the folders"
python3 - <<END
import os
import shutil
from data_preprocessing.paths import path_chest_imagenome, path_mimic_cxr, path_mimic_cxr_jpg
import subprocess
print("This is a Python command.")

def move_subfolder_to_parent_folder(subfolder_path, parent_folder_path):
    # move subfolder directly under parent folder as cut and paste
    shutil.move(subfolder_path, parent_folder_path)

move_subfolder_to_parent_folder(path_mimic_cxr_jpg + "/physionet.org/files/mimic-cxr-jpg/2.0.0/files", path_mimic_cxr_jpg)
END


echo "Fixing bounding boxes in the MIMIC-CXR dataset."
python3 - <<END
import shutil
from data_preprocessing.create import DataPreprocessing
shutil.copy('/content/drive/MyDrive/MIMIC/train_csv/train.csv','train.csv')
dataPreprocessor = DataPreprocessing()
dataPreprocessor.adjust_bounding_boxes("train.csv","new_train.csv")
# remove the old train.csv
os.remove("train.csv")
# rename the new train.csv to train.csv
os.rename("new_train.csv","train.csv")
# move the train.csv to datasets folder
shutil.move("train.csv", path_mimic_cxr)
END


echo "Done preprocessing"
