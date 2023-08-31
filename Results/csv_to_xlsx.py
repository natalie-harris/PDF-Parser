import os
import pandas as pd

def csv_to_xlsx(csv_path):
    # Check if the given path is a valid file
    if not os.path.isfile(csv_path):
        print(f"Error: File '{csv_path}' not found.")
        return

    # Read CSV into a DataFrame
    df = pd.read_csv(csv_path)

    # Generate the XLSX path by changing the file extension
    xlsx_path = os.path.splitext(csv_path)[0] + '.xlsx'

    # Save DataFrame to XLSX
    df.to_excel(xlsx_path, index=False)
    print(f"Successfully converted '{csv_path}' to '{xlsx_path}'.")

if __name__ == "__main__":
    csv_path = input("Enter the path to the CSV file: ")
    csv_to_xlsx(csv_path)
