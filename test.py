# Import Pandas
import pandas as pd

# Specify the name of the CSV file
csv_filename = r'Results\all_pdfs.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_filename)

df['relevance'] = 0

print(df)

df.to_csv(csv_filename, index=False)