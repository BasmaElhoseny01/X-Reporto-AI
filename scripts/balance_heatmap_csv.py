import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def convert_nulls_to_zeros(df):
    for column in df.columns:
        # check if the column name is not 'class'
        # if column == 'No Finding':
        #     continue
        # replace -1 with 1
        df[column] = df[column].replace(-1.0, 1.0)
        # 1. count the number of ones in the column
        num_ones = (df[column] == 1).sum()
        # 2. count the number of zeros in the column
        num_zeros = (df[column] == 0).sum()

        # 3. if the number of ones is greater than the number of zeros
        if num_ones > num_zeros:
            # 4. replace the nulls with zeros so that the number of zeros is equal to the number of ones
            diff = 0
            for i in range(len(df[column])):
                if pd.isnull(df[column][i]):
                    df[column][i] = 0
                    diff += 1
                if diff >= num_ones - num_zeros:
                    break        
        # 5. replaces nulls with zeros
        df[column] = df[column].replace(np.nan, -1.0)
    return df

def plot_histogram(df,dataset_description,plot=False,save=False):
    zeros=[]
    nulls=[]
    ones=[]
    negative=[]

    for column in df.columns:
        zeros.append((df[column]==0).sum())
        nulls.append( df[column].isnull().sum())
        ones.append((df[column]==1).sum())
        negative.append((df[column]==-1).sum())


    # make a histogram of the number of ones and zeros in each column
    fig, ax = plt.subplots(figsize=(12, 8))
    index = np.arange(len(df.columns))
    bar_width = 0.15
    opacity = 0.8
    plt.bar(index, ones, bar_width, alpha=opacity, color='b', label='ones')
    plt.bar(index + bar_width, zeros, bar_width, alpha=opacity, color='r', label='zeros')
    plt.bar(index + 2*bar_width, nulls, bar_width, alpha=opacity, color='g', label='nulls')
    plt.bar(index + 3*bar_width, negative, bar_width, alpha=opacity, color='y', label='negative')

    plt.xlabel('Columns')
    plt.ylabel('Frequency')
    plt.title('Frequency of Labels per class (' + dataset_description+')')
    plt.xticks(index + bar_width, df.columns, rotation=45)
    plt.legend()

    # Save the plot
    if save:
        plt.savefig("Histogram_"+dataset_description+".png")
    if plot:
        plt.show()

def balance_heatmap_csv(csv_path,plot,save):
    """
    Balances the heatmap CSV file by converting null values to zeros and saves the balanced CSV.

    Args:
        csv_path (str): Path to the input CSV file.
        plot (bool, optional): Whether to plot histograms before and after balancing. Defaults to False.
    """

    # Read CSV File
    df = pd.read_csv(csv_path)

    # get new dataframe with first 2 columns
    new_df = df.iloc[:, :2]

    # add last 2 columns of old dataframe to new dataframe
    new_df["image_id"] = df["image_id"]
    new_df["mimic_image_file_path"] = df["mimic_image_file_path"]


    # remove the first two columns
    # df = df.drop(df.columns[0], axis=1)
    # df = df.drop(df.columns[0], axis=1)

    # remove the last column
    # df = df.drop(df.columns[-1], axis=1)
    # df = df.drop(df.columns[-1], axis=1)

    df=df.drop(columns=["Consolidation","Enlarged Cardiomediastinum","Fracture","Lung Lesion","Pleural Other","Pneumothorax"])
    # df=df.drop(columns=["Enlarged Cardiomediastinum","Fracture","Lung Lesion","Pleural Other"])

    if plot or save:
        plot_histogram(df,"Before_Balancing",plot=plot,save=save)



    # convert nulls to zeros
    df = convert_nulls_to_zeros(df)

    # append df to new_df
    new_df = pd.concat([new_df, df], axis=1)

    # plot histogram
    if plot or save:
        plot_histogram(df,"After_Balancing",plot=plot,save=save)

    # save new_df to csv
    output_path=csv_path.replace(".csv","_balanced.csv")
    new_df.to_csv(output_path, index=False)
    print("Balanced CSV saved to", output_path)



# Script to balance the heatmap csv file
if __name__ == "__main__":
    # Take csv path as argument
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, help='Path to the csv file')
    parser.add_argument('--plot', action="store_true", help='Plot histogram of the labels')
    parser.add_argument('--save', action="store_true", help='Save histogram of the labels')

    args = parser.parse_args()

    balance_heatmap_csv(args.csv_path,args.plot,args.save)

    print("Heatmap CSV balanced successfully!")

# Usage:
# python ./scripts/balance_heatmap_csv.py --csv_path "./datasets/heat_map_train.csv"
# python ./scripts/balance_heatmap_csv.py --csv_path "./datasets/heat_map_train.csv"  --plot
# python ./scripts/balance_heatmap_csv.py --csv_path "./datasets/heat_map_train.csv"  --save