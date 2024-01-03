import os
import shutil
from paths import path_chest_imagenome, path_mimic_cxr, path_mimic_cxr_jpg
# create folders in paths if they don't exist
def create_folders(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

create_folders([path_chest_imagenome, path_mimic_cxr, path_mimic_cxr_jpg])

# unzip the mimic-cxr-jpg dataset
!unzip "/content/drive/MyDrive/MIMIC/p11_subset1.zip" -d $path_mimic_cxr_jpg
!unzip "/content/drive/MyDrive/MIMIC/chest-imagenome-dataset-1.0.0.zip" -d $path_chest_imagenome
!unzip $path_chest_imagenome+"/silver_dataset/scene_graph.zip" -d $path_chest_imagenome+"/silver_dataset/
!unzip "/content/drive/MyDrive/MIMIC/mimic-cxr-reports.zip" -d $path_mimic_cxr

# in folder mimic-cxr-jpg, move subfolder directly under mimic-cxr-jpg

def move_subfolder_to_parent_folder(subfolder_path, parent_folder_path):
    # move subfolder directly under parent folder as cut and paste
    shutil.move(subfolder_path, parent_folder_path)
    
move_subfolder_to_parent_folder(path_mimic_cxr_jpg + "/physionet.org/files/mimic-cxr-jpg/files", path_mimic_cxr_jpg)




