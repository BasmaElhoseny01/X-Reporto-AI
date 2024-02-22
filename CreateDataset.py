import shutil
import os

def main():
    ''''
    # /home/basma/Desktop/csv_processing/p10_subset1/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10
    #                                                            datasets/mimic-cxr-jpg/files/p10/
    '''

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
    # Call the main function
    main()