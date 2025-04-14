import pandas as pd

# Load data
df = pd.read_csv("/projects/federated_learning/Accenture/BreastCancerDataRoystonAltman/breast-cancer-data-royston-altman.csv")
target_col = "status"
age_col = "age"  # update this if your age column has a different name

# Sort by age
df_sorted = df.sort_values(by=age_col).reset_index(drop=True)

# Separate by recurrence
df_recur = df_sorted[df_sorted[target_col] == 1].reset_index(drop=True)
df_non_recur = df_sorted[df_sorted[target_col] != 1].reset_index(drop=True)

# Split recurrence by age (younger to A, older to B)
n_recur = len(df_recur)
mid_recur = n_recur // 2
df_recur_A = df_recur.iloc[:mid_recur + (n_recur % 2)]
df_recur_B = df_recur.iloc[mid_recur + (n_recur % 2):]

# Split non-recurrence by age (younger to A, older to B)
n_non_recur = len(df_non_recur)
mid_non_recur = n_non_recur // 2
df_non_recur_A = df_non_recur.iloc[:mid_non_recur + (n_non_recur % 2)]
df_non_recur_B = df_non_recur.iloc[mid_non_recur + (n_non_recur % 2):]

# Combine subsets
subset_A = pd.concat([df_recur_A, df_non_recur_A]).reset_index(drop=True)
subset_B = pd.concat([df_recur_B, df_non_recur_B]).reset_index(drop=True)

# Optionally shuffle each subset
subset_A = subset_A.sample(frac=1, random_state=42).reset_index(drop=True)
subset_B = subset_B.sample(frac=1, random_state=42).reset_index(drop=True)

# Print recurrence stats
print("Subset A recurrence:")
print(subset_A[target_col].value_counts())
print("\nSubset B recurrence:")
print(subset_B[target_col].value_counts())

#Print age stats per split dataset
print("\nSubset A age stats:")
print(subset_A[age_col].describe())
print("\nSubset B age stats:")
print(subset_B[age_col].describe())

# Save subsets
subset_A.to_csv("BreastCancerDataRoystonAltman_subset_A_younger.csv", index=False)
subset_B.to_csv("BreastCancerDataRoystonAltman_subset_B_older.csv", index=False)

print("\nSubsets saved as 'BreastCancerDataRoystonAltman_subset_A_younger.csv' and 'BreastCancerDataRoystonAltman_subset_B_older.csv'")
