import pandas as pd

# Load the existing CSV file
df = pd.read_csv('data.csv')

# Filter out rows with "Unknown" label
df_filtered = df[df['label'] != "Unknown"]

# Save the filtered data to a new CSV file
df_filtered.to_csv('data_filtered.csv', index=False)

print(f"Original data had {len(df)} entries")
print(f"Filtered data has {len(df_filtered)} entries")
print(f"Removed {len(df) - len(df_filtered)} entries with 'Unknown' label")