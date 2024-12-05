import pandas as pd

# Load the dataset
file_path = 'testing_dataset.csv'  # Replace with your dataset file path
dataset = pd.read_csv(file_path)

# Define labeling rules
def label_ddos(row):
    # Rule: If 'Flow Packets/s' > 1000 or 'SYN Flag Count' > 10, label as 'DDoS'
    if row['Flow Packets/s'] > 1000 or row['SYN Flag Count'] > 10:
        return 'DDoS'
    else:
        return 'Normal'

# Apply the labeling function to the dataset
dataset['Auto_Label'] = dataset.apply(label_ddos, axis=1)

# Count the number of 'DDoS' labels
ddos_count = dataset['Auto_Label'].value_counts().get('DDoS', 0)

# Print the count of 'DDoS' labels
print(f"Number of records labeled as 'DDoS': {ddos_count}")

# Save the labeled dataset
output_file_path = 'labeled_dataset.csv'  # Output file name
dataset.to_csv(output_file_path, index=False)

print(f"Labeled dataset saved to {output_file_path}")
