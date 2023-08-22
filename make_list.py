import os
import pandas as pd

directories = [
    r'C:\Users\natal\OneDrive\Documents\GitHub\chatgpt_cost_estimate\papers',
    r'E:\NIMBioS\SBW\SBW Literature\Canadian Entomologist',
    r'E:\NIMBioS\SBW\SBW Literature\Google Scholar',
    r'E:\NIMBioS\SBW\SBW Literature\Nature Portfolio',
    r'E:\NIMBioS\SBW\SBW Literature\Springer Link\new_pdfs',
    r'E:\NIMBioS\SBW\SBW Literature\Canada Government\pdfs',
    r'E:\NIMBioS\SBW\SBW Literature\References Finder'
]

file_names = []

# Collect all pdfs from directories
for dir in directories:
    for file in os.listdir(dir):
        if file.endswith('.pdf') and not file.startswith('.'):
            full_path = os.path.join(dir, file)
            file_names.append(full_path)

print(file_names)

df = pd.DataFrame({
    'file_name': file_names,
    'been_processed': [0] * len(file_names)  # List of zeros, same length as file_names
})

csv_filename = 'all_pdfs.csv'

df.to_csv(csv_filename, index=False)

