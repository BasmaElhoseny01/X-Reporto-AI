# logger_setup.py
import logging
import datetime
import os

def setup_logging(log_file_path,bash=False,periodic_logger=False,log_folder_path=None):
    if log_folder_path is None: log_folder_path='logs/'

    if not os.path.exists(log_folder_path):
        os.makedirs(log_folder_path)
        print(f"Folder {log_folder_path} created successfully.")

    # Configure logging
    logging.basicConfig(filename=log_file_path ,encoding='utf-8', level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    print("logging at ",log_file_path)
    
    if periodic_logger:
        # Periodic Logging
        five_days_log_file_path=five_day_logger_handler()
        # Add a second file handler to log to the second file
        periodic_file_handler = logging.FileHandler(five_days_log_file_path, encoding='utf-8')
        periodic_file_handler.setLevel(logging.DEBUG)  # Set the desired level for the second file
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        periodic_file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(periodic_file_handler)
    
    if bash:
        # Create a StreamHandler to output log messages to the terminal
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)  # Set the desired level for console output

        # Create a formatter and set it for both file and console handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add the console handler to the root logger
        logging.getLogger().addHandler(console_handler)

    # Log start marker with current date and time
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f'=================== New Run Started: {current_time} ===================')


def five_day_logger_handler():
    five_days_log_folder_path='./logs/5-days/'

    if not os.path.exists(five_days_log_folder_path):
        os.makedirs(five_days_log_folder_path)
        print(f"Folder '{five_days_log_folder_path}' created successfully.")
    
    log_files = [file for file in os.listdir(five_days_log_folder_path) if file.endswith('.log')]

    if log_files:
        # Sort the log files based on their names (which include the date)
        sorted_log_files = sorted(log_files, reverse=True)

        # Get the newest log file
        newest_log_file = sorted_log_files[0]
        # print(f"The newest log file is: {newest_log_file}")

        # Extract the creation date from the newest log file name
        creation_date_str = newest_log_file.split('_')[0]
        creation_date = datetime.datetime.strptime(creation_date_str, '%Y-%m-%d')
    else:
        creation_date=None


    # Get current date and time
    current_date = datetime.datetime.now()

    # Check if 5 days have passed since the creation of the newest log file
    if creation_date is None or current_date - creation_date >= datetime.timedelta(days=1):
        # print("5 days have passed since the creation of the newest log file.")

        # Get current date and time
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H-%M-%S')

        # Concatenate date and time with underscore
        file_name = f"{current_date}_{current_time}.log"
        
        five_days_log_file_path=five_days_log_folder_path+'/'+file_name

        with open(five_days_log_file_path, 'w') as file:
            file.write(f"New 5-day Log started at {current_date} \n\n")


    else:
        # print("Less than 5 days have passed since the creation of the newest log file.")
        five_days_log_file_path=five_days_log_folder_path+'/'+newest_log_file


    print("5-day logging at",five_days_log_file_path)
    return five_days_log_file_path


if __name__ == "__main__":
    setup_logging()

    # five_days_log_file_path=five_day_logger_handler()
    # print(five_days_log_file_path)

    # Get list of log files
    # five_days_log_folder_path='./logs/5-days/'
