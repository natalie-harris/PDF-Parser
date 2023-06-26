import pandas as pd

# Define the function
def build_dataframe(df):
    # Step 1: Sort the DataFrame by 'area' and then by 'Year'
    df = df.sort_values(['area', 'Year'])
    
    # Step 2: Fill in missing years with 'Outbreak' = 0
    filled_rows = []
    for area, group in df.groupby('area'):
        last_outbreak_year = None
        last_outbreak_value = None
        for _, row in group.iterrows():
            if row['Outbreak'] in [1, 2]:
                if (last_outbreak_value is not None) and (row['Year'] >= last_outbreak_year + 2) and (row['Outbreak'] in [1, 2]):
                    filled_row = row.copy()
                    filled_row['Outbreak'] = 0
                    for i in range(int(last_outbreak_year) + 1, int(row['Year'])):
                        print(filled_row['Outbreak'])
                        filled_row['Year'] = i
                        filled_rows.append(filled_row)
                last_outbreak_year = row['Year']
                last_outbreak_value = row['Outbreak']
    
    df = pd.concat([df, pd.DataFrame(filled_rows)], ignore_index=True).sort_values(['area', 'Year'])
    
    # Step 3: Handle entries with matching 'Latitude', 'Longitude', and 'Year'
    def combine_outbreaks(group):
        outbreak_value = 1 if group['Outbreak'].eq(1).any() else 2
        combined_row = group.iloc[0].copy()
        combined_row['Outbreak'] = outbreak_value
        return combined_row
    
    df = (df.groupby(['Latitude', 'Longitude', 'Year'], as_index=False)
          .apply(combine_outbreaks)
          .reset_index(drop=True))
    
    return df

# Create an example DataFrame
data = {
    'area': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C'],
    'Year': [2020, 2021, 2023, 2020, 2022, 2023, 2022, 2023],
    'Outbreak': [1, 1, 1, 2, 2, 2, 1, 1],
    'Latitude': [34, 34, 34, 50, 50, 50, 45, 45],
    'Longitude': [-118, -118, -118, -122, -122, -122, -80, -80]
}

df = pd.DataFrame(data)

# Apply the build_dataframe function
modified_df = build_dataframe(df)

# Display the modified DataFrame
print(modified_df)
