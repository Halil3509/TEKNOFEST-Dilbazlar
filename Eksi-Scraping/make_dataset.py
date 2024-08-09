import pandas as pd

# Load the CSV file into a DataFrame
combined_df = pd.read_csv("filtered_entries1.csv")

# Filter the DataFrame to keep only the rows where the "Result" column is "yes"
filtered_df = combined_df[combined_df["Result"] == "yes"]

# Define a list of labels to be combined into a single DataFrame
combine_labels = ["depresyon", "depresyon belirtileri"]

# Define a list of labels for separate DataFrames
other_labels = ["distimi"]

# Filter the DataFrame to include only rows with titles in the combine_labels list
combined_df = filtered_df[filtered_df["Title"].isin(combine_labels)]

# Create a dictionary of DataFrames for each label in other_labels
individual_dfs = {label: filtered_df[filtered_df["Title"] == label] for label in other_labels}

# Save the combined DataFrame to a CSV file
combined_df.to_csv("depresyon.csv", index=False)

# Save each individual DataFrame to a separate CSV file
for label, group in individual_dfs.items():
    group.to_csv(f"{label}.csv", index=False)
