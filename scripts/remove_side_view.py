import os
import argparse
import pandas as pd

def main(data_set_path,csv_file_path):

    # Read CSV file
    data_csv=pd.read_csv(csv_file_path,usecols=['subject_id','study_id','image_id'])


    # List all folders in the directory
    folders = [folder for folder in os.listdir(data_set_path) if os.path.isdir(os.path.join(data_set_path, folder))]

    # list kept cases 
    kept_cases = []
    # List of Completely RemOved Cases
    removed_cases = []

    
    # Print the list of folders
    print("List of folders:")
    for folder in folders:
        print(folder)
        # ie:P10 folders
        for subject in os.listdir(os.path.join(data_set_path, folder)):
            if os.path.isdir(os.path.join(data_set_path, folder, subject)):
            # ie:P10992759 folders
            # Check if the patient folder is a directory
                # print(subject)

                for study_folder in os.listdir(os.path.join(data_set_path, folder, subject)):
                    # ie:S10416870 folders
                    # Check if the study folder is a directory
                    if os.path.isdir(os.path.join(data_set_path, folder, subject, study_folder)):
                        # print(" "+study_folder)
                        keep_case = False
                        for image in os.listdir(os.path.join(data_set_path, folder, subject, study_folder)):
                            # Check if the image is a file
                            if os.path.isfile(os.path.join(data_set_path, folder, subject, study_folder, image)):
                                # print("     "+image)
                                # Get the image file path
                                image_id = image.split(".")[0]
                                image_ext = image.split(".")[1]
                                # print("subject",subject[1:],int(subject[1:]) not in data_csv['subject_id'].values)
                                # return 
                                # print("study_folder",study_folder[1:], int(study_folder[1:]) not in data_csv['study_id'].values)
                                # print("image_id",image_id,image_id not in data_csv['image_id'].values)

                                if image_ext!="html" and (int(subject[1:]) not in data_csv['subject_id'].values or int(study_folder[1:]) not in data_csv['study_id'].values or image_id not in data_csv['image_id'].values):
                                    img_file_path = os.path.join(data_set_path, folder, subject, study_folder, image)
                                    img_file_path = img_file_path.replace("\\", "/")
                                    os.remove(img_file_path)
                                    print(f"Removed {img_file_path}")
                                else:
                                    keep_case = True
                                    # print(f"Kept {folder}/{subject}/{study_folder}/{image}")


                        if keep_case:
                            kept_cases.append(f"{folder}/{subject}/{study_folder}")
                        else:
                            removed_cases.append(f"{folder}/{subject}/{study_folder}")

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

    args = parser.parse_args()
    # Call the main function with the provided file path argument
    main(data_set_path=args.data_set,csv_file_path=args.csv)

# python ./scripts/remove_side_view.py --csv ./datasets/heat_map_train_final.csv --data_set ./datasets_try/mimic-cxr-jpg/files