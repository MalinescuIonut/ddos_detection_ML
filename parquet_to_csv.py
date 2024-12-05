import pandas as pd

# Load the Parquet file
parquet_file_path = 'Syn-testing.parquet'  # Replace with the actual file path
df = pd.read_parquet(parquet_file_path)

# Save as a CSV file
csv_file_path = 'testing_dataset.csv'  # Specify the output CSV file path
df.to_csv(csv_file_path, index=False)

print(f"Parquet file has been saved as CSV at {csv_file_path}")
