import pandas as pd

# Load data
df = pd.read_csv("/projects/federated_learning/Accenture/BreastCancerDataRoystonAltman/breast-cancer-data-royston-altman.csv")
age_col = "age"  # update if your column is named differently
target_col = "status"

# Calculate overall median age
median_age = df[age_col].median()
print(median_age)

# Split into two groups based on median
subset_A = df[df[age_col] < median_age].reset_index(drop=True)
subset_B = df[df[age_col] >= median_age].reset_index(drop=True)

# Optionally shuffle
subset_A = subset_A.sample(frac=1, random_state=42).reset_index(drop=True)
subset_B = subset_B.sample(frac=1, random_state=42).reset_index(drop=True)

# Check distributions
print("Subset A (younger):")
print(subset_A[age_col].describe())
print(subset_A[target_col].value_counts())

print("\nSubset B (older):")
print(subset_B[age_col].describe())
print(subset_B[target_col].value_counts())

# Save to file
subset_A.to_csv("BreastCancer_subset_A_younger.csv", index=False)
subset_B.to_csv("BreastCancer_subset_B_older.csv", index=False)

print("\nSaved: BreastCancer_subset_A_younger.csv and BreastCancer_subset_B_older.csv")
