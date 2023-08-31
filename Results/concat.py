import glob
import pandas as pd
import os

def wait():
    input("Waiting...")

def combine_csv_files(folder_path):
    # Use glob to get all the csv files in the folder
    all_csv_files = glob.glob(f"{folder_path}/*.csv")

    # Filter out files that don't have a number in the filename
    csv_files_with_numbers = [file for file in all_csv_files if any(char.isdigit() for char in os.path.basename(file))]

    # Read and combine all the CSV files with numbers in their filenames
    combined_data = pd.concat([pd.read_csv(file) for file in csv_files_with_numbers])

    # Save the combined data to 'all_data.csv'
    combined_data.to_csv(f"{folder_path}/all_data.csv", index=False)

if __name__ == "__main__":
    folder_path = '/Users/natalieharris/UTK/NIMBioS/Spruce Budworms/Parser 2/Results'
    combine_csv_files(folder_path)
    print(f"All data combined and saved to {folder_path}/all_data.csv")
