import pandas as pd

# Path to the original CSV file
original_file_path = "processed_data.csv"

# Path to save the randomly selected rows
new_file_path = "random_sample_data.csv"

# Read the original CSV file
data = pd.read_csv(original_file_path)

# Get the number of rows in the dataset
num_rows = data.shape[0]

# Define the number of rows to randomly select
num_rows_to_select = 5000

# Check if the dataset has at least 5000 rows
if num_rows <= num_rows_to_select:
    print("Error: The dataset does not have enough rows.")
else:
    # Randomly select 5000 row indices
    random_indices = data.sample(n=num_rows_to_select, random_state=42).index

    # Select the rows corresponding to the random indices
    random_data = data.loc[random_indices]

    # Save the randomly selected data to a new CSV file
    random_data.to_csv(new_file_path, index=False)

    print("Randomly selected data has been saved to", new_file_path)
