import pandas as pd
df = pd.read_csv("/projects/federated_learning/Accenture/BreastCancerDataRoystonAltman/breast-cancer-data-royston-altman.csv")
target_col = "status"
df_recur = df[df[target_col] == 1].sample(frac=1, random_state=42)
df_non_recur = df[df[target_col] != 1].sample(frac=1, random_state=42)
n_recur = len(df_recur)
mid_recur = n_recur // 2
# If the number of recurrent cases is odd, give  extra case to the first subset.
df_recur_A = df_recur.iloc[:mid_recur + (n_recur % 2)]
df_recur_B = df_recur.iloc[mid_recur + (n_recur % 2):]

# Split non-recurrent into 2 equal parts.
n_non_recur = len(df_non_recur)
mid_non_recur = n_non_recur // 2
df_non_recur_A = df_non_recur.iloc[:mid_non_recur + (n_non_recur % 2)]
df_non_recur_B = df_non_recur.iloc[mid_non_recur + (n_non_recur % 2):]

# Combine splits in 2 subsets.
subset_A = pd.concat([df_recur_A, df_non_recur_A]).sample(frac=1, random_state=42).reset_index(drop=True)
subset_B = pd.concat([df_recur_B, df_non_recur_B]).sample(frac=1, random_state=42).reset_index(drop=True)

print("Subset A recurrence:")
print(subset_A[target_col].value_counts())
print("\nSubset B recurrence :")
print(subset_B[target_col].value_counts())

# Save 
subset_A.to_csv("BreastCancerDataRoystonAltman_subset_A.csv", index=False)
subset_B.to_csv("BreastCancerDataRoystonAltman_subset_B.csv", index=False)

print("\nSubsets saved as 'BreastCancerDataRoystonAltman_subset_A.csv' and 'BreastCancerDataRoystonAltman_subset_B.csv'")
