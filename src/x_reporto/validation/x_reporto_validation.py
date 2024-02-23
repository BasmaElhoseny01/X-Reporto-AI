# Logging
from logger_setup import setup_logging
import logging


import os
import gc

# Torch
import torch
import torch.optim as optim
from torch.utils.data import  DataLoader

# Modules
from src.x_reporto.models.x_reporto_factory import XReporto
from src.x_reporto.data_loader.custom_dataset import CustomDataset

# Utils 

from config import RUN,PERIODIC_LOGGING,log_config
from config import *


class XReportoValidation():
    def __init__(self, model:XReporto,validation_csv_path:str = validation_csv_path):
        pass
    def validate(self):
        # make model in training mode
        logging.info("Start Validation")
        pass


def main():
    logging.info(" X_Reporto Started")
    # Logging Configurations
    log_config()

   
    # X-Reporto Trainer Object
    x_reporto_model = XReporto().create_model()

    # Create an XReportoTrainer instance with the X-Reporto model
    validator = XReportoValidation(model=x_reporto_model)


    # # Start Training
    validator.validate()
        

if __name__ == '__main__':
    # Call the setup_logging function at the beginning of your script
    setup_logging(log_file_path='./logs/x_reporto_validator.log',bash=True,periodic_logger=PERIODIC_LOGGING)

    try:
        # The main script runs here
        main()
    except Exception as e:
        # Log any exceptions that occur
        logging.exception("An error occurred",exc_info=True)
    

     