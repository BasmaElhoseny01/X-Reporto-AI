import os
import pandas as pd

def main(data_set_path,csv_file_path):

    # Read CSV file
    data_csv=pd.read_csv(csv_file_path)
    data_csv.top(10)
    return 

    # List all folders in the directory
    folders = [folder for folder in os.listdir(data_set_path) if os.path.isdir(os.path.join(data_set_path, folder))]
    
    # Print the list of folders
    print("List of folders:")
    for folder in folders:
        # print(folder)
        # ie:P10 folders
        for patient_folder in os.listdir(os.path.join(data_set_path, folder)):
            if os.path.isdir(os.path.join(data_set_path, folder, patient_folder)):
            # ie:P10992759 folders
            # Check if the patient folder is a directory
                # print(patient_folder)

                for study_folder in os.listdir(os.path.join(data_set_path, folder, patient_folder)):
                    # Check if the study folder is a directory
                    if os.path.isdir(os.path.join(data_set_path, folder, patient_folder, study_folder)):
                        print(study_folder)
                        for image in os.listdir(os.path.join(data_set_path, folder, patient_folder, study_folder)):
                            # Check if the image is a file
                            if os.path.isfile(os.path.join(data_set_path, folder, patient_folder, study_folder, image)):
                                print(image)
                                # Check if this file is in the dataset csv file
                                # if image not in dataset_files:
                                #     # Remove the image
                                #     os.remove(os.path.join(data_set_path, folder, patient_folder, study_folder, image))
                                #     print(f"Removed {image} from {folder}/{patient_folder}/{study_folder}")

                                # # Check if the patient folder contains the word "view"
                                # if "view" in image:
                                #     # Remove the patient folder
                                #     os.remove(os.path.join(data_set_path, folder, patient_folder, study_folder, image))
                                #     print(f"Removed {image} from {folder}/{patient_folder}/{study_folder}")


                # Check if the patient folder contains the word "view"
                # if "view" in patient_folder:
                #     # Remove the patient folder
                #     os.rmdir(os.path.join(data_set_path, folder, patient_folder))
                #     print(f"Removed {patient_folder} from {folder}")
        print(folder)


if __name__ == '__main__':
    main(data_set_path="./datasets_test/mimic-cxr-jpg/files",csv_file_path='./datasets/train_full.csv')