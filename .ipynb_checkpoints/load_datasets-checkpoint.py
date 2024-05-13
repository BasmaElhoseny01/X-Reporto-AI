import os
import shutil
from paths import path_chest_imagenome, path_mimic_cxr, path_mimic_cxr_jpg
import subprocess
# create folders in paths if they don't exist
def create_folders(paths):
    print("start")
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print("created successfully path: ",path)

create_folders([path_chest_imagenome, path_mimic_cxr, path_mimic_cxr_jpg])

# unzip the mimic-cxr-jpg dataset
# !unzip "/content/drive/MyDrive/MIMIC/p11_subset1.zip" -d "datasets/mimic-cxr-jpg"
subprocess.run(["unzip", "/content/drive/MyDrive/MIMIC/p11_subset1.zip" , "-d", path_mimic_cxr_jpg], check=True, shell=True)
print("unzipped mimic_cxr_jpg")
# !unzip "/content/drive/MyDrive/MIMIC/chest-imagenome-dataset-1.0.0.zip" -d "datasets/chest-imagenome-dataset"
subprocess.run(["unzip", "/content/drive/MyDrive/MIMIC/chest-imagenome-dataset-1.0.0.zip" , "-d", path_chest_imagenome], check=True, shell=True)
print("unzipped chest imagnome")
# !unzip "datasets/chest-imagenome-dataset/silver_dataset/scene_graph.zip" -d "datasets/chest-imagenome-dataset/silver_dataset/
subprocess.run(["unzip", "datasets/chest-imagenome-dataset/silver_dataset/scene_graph.zip" , "-d", "datasets/chest-imagenome-dataset/silver_dataset/"], check=True, shell=True)
print("unzipped scene_graph")
# !unzip "/content/drive/MyDrive/MIMIC/mimic-cxr-reports.zip" -d "datasets/mimic-cxr"
subprocess.run(["unzip","/content/drive/MyDrive/MIMIC/mimic-cxr-reports.zip" , "-d", path_mimic_cxr], check=True, shell=True)

# in folder mimic-cxr-jpg, move subfolder directly under mimic-cxr-jpg

def move_subfolder_to_parent_folder(subfolder_path, parent_folder_path):
    # move subfolder directly under parent folder as cut and paste
    shutil.move(subfolder_path, parent_folder_path)
    
# wait for unzip subprocess to finish
for i in range(100000000):
    if i % 10000000 == 0:
        print(i/10000000 *10 , "%")
    pass
move_subfolder_to_parent_folder(path_mimic_cxr_jpg + "/physionet.org/files/mimic-cxr-jpg/2.0.0/files", path_mimic_cxr_jpg)



# rename mimic-cxr-jpg to mimic-cxr-jpg-2.0.0
os.rename(path_mimic_cxr_jpg + "/files", path_mimic_cxr_jpg + "/2.0.0")
