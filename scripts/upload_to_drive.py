from google.oauth2 import service_account
from googleapiclient.discovery import build
import os

# Path to the service account key file (JSON format)
KEY_FILE = '/path/to/service_account_key.json'

# Authenticate and create the Drive service
credentials = service_account.Credentials.from_service_account_file(KEY_FILE, scopes=['https://www.googleapis.com/auth/drive'])
drive_service = build('drive', 'v3', credentials=credentials)

def upload_file(file_path, folder_id=None):
    file_name = os.path.basename(file_path)
    
    # Metadata for the file
    metadata = {'name': file_name}
    if folder_id:
        metadata['parents'] = [folder_id]

    # Upload the file
    media = {'media': open(file_path, 'rb')}
    file = drive_service.files().create(body=metadata, media_body=media, fields='id').execute()

    print(f'File uploaded: {file_name} (ID: {file["id"]})')

# Example usage
upload_file('/path/to/local/file.txt')  # Upload file to root folder
# or
upload_file('/path/to/local/file.txt', folder_id='your_folder_id')  # Upload file to a specific folder
