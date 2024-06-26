"""
Description:
------------
This script reads a HeatMap checkpoint and updates its optimal thresholds based on the new optimal thresholds provided.

Usage:
------
To run this script, execute the following command from the terminal:

    python -m scripts.update_optimal_th_in_checkpoint --model_path <path_to_model_checkpoint> --optimal_thresholds_path <path_to_optimal_thresholds_file> --new_model_path <path_to_save_updated_model_checkpoint>

Example:
--------
To run the script, execute:

    >>> python -m scripts.update_optimal_th_in_checkpoint --model_path ./models/heat_map_4/heat_map_best.pth --optimal_thresholds_path ./models/heat_map_4/optimal_thresholds.txt --new_model_path ./models/heat_map_4/heat_map_best_op_th.pth

Directory:
----------
This script should be located in the `/Graduation-Project` directory within the project.
"""

import argparse
import torch

from src.heat_map_U_ones.models.heat_map import HeatMap

def read_optimal_thresholds(optimal_thresholds_path:str):
    # Read the optimal thresholds
    optimal_thresholds = []
    with open(optimal_thresholds_path, "r") as f:
        for line in f:
            optimal_thresholds.append(float(line.strip()))
    return optimal_thresholds

def main(model_path:str,optimal_thresholds_path:str,new_model_path:str):
    # Heat Map Model 
    heat_map_model = HeatMap()

    # Read the checkpoint
    print("Loading heat_map ....")
    heat_map_model.load_state_dict(torch.load(model_path))

    old_optimal_thresholds = heat_map_model.optimal_thresholds

    # Read the new optimal thresholds
    new_optimal_thresholds = read_optimal_thresholds(optimal_thresholds_path)
    
    # Update the optimal thresholds
    heat_map_model.optimal_thresholds = new_optimal_thresholds

    print(f"Old optimal thresholds: {old_optimal_thresholds}")
    print(f"New optimal thresholds: {heat_map_model.optimal_thresholds}")

    # Save the updated model
    torch.save(heat_map_model.state_dict(), new_model_path)
    print("Model saved successfully at: ",new_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update optimal thresholds in HeatMap CheckPoint")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--optimal_thresholds_path", type=str, required=True, help="Path to the optimal thresholds file")
    parser.add_argument("--new_model_path", type=str, required=True, help="Path to save the new model checkpoint with updated thresholds")
    
    args = parser.parse_args()

    model_path = args.model_path
    optimal_thresholds_path = args.optimal_thresholds_path
    new_model_path = args.new_model_path

    print(f"Model path: {model_path}")
    print(f"Optimal thresholds path: {optimal_thresholds_path}")
    print(f"New model path: {new_model_path}")

    # Call the main function
    # main(model_path="./models/heat_map_4/heat_map_best.pth",optimal_thresholds_path="./models/heat_map_4/optimal_thresholds.txt",new_model_path="./models/heat_map_4/heat_map_best_op_th.pth")
    main(model_path=model_path,optimal_thresholds_path=optimal_thresholds_path,new_model_path=new_model_path)


# PS D:\Graduation-Project> python -m scripts.update_optimal_th_in_checkpoint --model_path ./models/heat_map_4/heat_map_best.pth --optimal_thresholds_path ./models/heat_map_4/optimal_thresholds.txt --new_model_path ./models/heat_map_4/heat_map_best_op_th.pth
