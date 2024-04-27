import os
import argparse
import pandas as pd

def main(data_set_path,csv_file_path,base_data_set="datasets/"):

    # Read CSV file
    data_csv=pd.read_csv(csv_file_path)['mimic_image_file_path'].tolist()

    # Replace backslashes with forward slashes in data_csv
    for i in range(len(data_csv)):
        data_csv[i] = data_csv[i].replace("\\", "/")
        # Replace datasets/ with datasets_test/
        data_csv[i] = data_csv[i].replace("datasets/", base_data_set)

    # List all folders in the directory
    folders = [folder for folder in os.listdir(data_set_path) if os.path.isdir(os.path.join(data_set_path, folder))]

    # list kept cases 
    kept_cases = []
    # List of Completely RemOved Cases
    removed_cases = []

    
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
                    # ie:S10416870 folders
                    # Check if the study folder is a directory
                    if os.path.isdir(os.path.join(data_set_path, folder, patient_folder, study_folder)):
                        # print(study_folder)
                        keep_case = False
                        for image in os.listdir(os.path.join(data_set_path, folder, patient_folder, study_folder)):
                            # Check if the image is a file
                            if os.path.isfile(os.path.join(data_set_path, folder, patient_folder, study_folder, image)):
                                # print(image)
                                # Get the image file path
                                img_file_path = os.path.join(data_set_path, folder, patient_folder, study_folder, image)
                                # Replace backslashes with forward slashes
                                img_file_path = img_file_path.replace("\\", "/")

                                if img_file_path[0:2] == "./":
                                    img_file_path = img_file_path[2:]
                                # print(data_csv)
                                # return

                                # check if img_file_path is an image file
                                if img_file_path not in data_csv:
                                # and (img_file_path[-4:] == ".jpg" or img_file_path[-4:] == ".png" or img_file_path[-5:] == ".jpeg" or img_file_path[-4:] == ".gif"):
                                    # Check if the image is in the CSV file
                                    # Remove the image
                                    # print(data_csv)
                                    # print(img_file_path)
                                    # os.remove(img_file_path)
                                    print(f"Removed {image} from {folder}/{patient_folder}/{study_folder}")
                                    # return 
                                else:
                                    keep_case = True
                                    print(f"Kept {image} from {folder}/{patient_folder}/{study_folder}")
                                    


                        if keep_case:
                            kept_cases.append(f"{folder}/{patient_folder}/{study_folder}")
                        else:
                            removed_cases.append(f"{folder}/{patient_folder}/{study_folder}")

    # Save the kept and removed cases to a text file
    with open("kept_cases.txt", "w") as f:
        for case in kept_cases:
            f.write(case + "\n")
    with open("removed_cases.txt", "w") as f:
        for case in removed_cases:
            f.write(case + "\n")
    # Print the number of kept and removed cases
    print(f"Kept cases: {len(kept_cases)}")
    print(f"Removed cases: {len(removed_cases)}")
            
        



                    
if __name__ == '__main__':
    # Take arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to the CSV file")
    parser.add_argument("--data_set", required=True, help="Path to the dataset")
    parser.add_argument("--base_data_set", required=False, help="Path to the base dataset", default="datasets/")

    args = parser.parse_args()
    # Call the main function with the provided file path argument
    # main(data_set_path="./datasets_test/mimic-cxr-jpg/files",csv_file_path='./datasets/train_full.csv',base_data_set="datasets_test/")
    main(data_set_path=args.data_set,csv_file_path=args.csv,base_data_set=args.base_data_set)

# python remove_side_view.py --csv ./datasets/heat_map_train_final.csv --data_set ./datasets_try/mimic-cxr-jpg/files --base_data_set datasets_try/