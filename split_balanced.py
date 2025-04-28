import pandas as pd
from sklearn.utils import resample

# Optional advanced balancing
# from imblearn.over_sampling import SMOTE, RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler

# Load your dataset
df = pd.read_csv("/projects/federated_learning/Accenture/BreastCancerDataRoystonAltman/breast-cancer-data-royston-altman.csv")
target_col = "status"

# Separate classes
df_recur = df[df[target_col] == 1]
df_non_recur = df[df[target_col] != 1]

# Method 1: Undersampling (Downsample the majority class)
# -----------------------------------------------
# min_class_size = min(len(df_recur), len(df_non_recur))
# df_recur = df_recur.sample(n=min_class_size, random_state=42)
# df_non_recur = df_non_recur.sample(n=min_class_size, random_state=42)
# df_balanced = pd.concat([df_recur, df_non_recur])

# Method 2: Oversampling (Duplicate the minority class)
# -----------------------------------------------
# max_class_size = max(len(df_recur), len(df_non_recur))
# if len(df_recur) < len(df_non_recur):
#     df_recur = resample(df_recur, replace=True, n_samples=max_class_size, random_state=42)
# else:
#     df_non_recur = resample(df_non_recur, replace=True, n_samples=max_class_size, random_state=42)
# df_balanced = pd.concat([df_recur, df_non_recur])

# Method 3: SMOTE (Synthetic oversampling for numeric features only)
# -----------------------------------------------
# df_encoded = pd.get_dummies(df.drop(columns=[target_col]))  # Use only if you have categorical vars
# smote = SMOTE(random_state=42)
# X_res, y_res = smote.fit_resample(df_encoded, df[target_col])
# df_balanced = pd.DataFrame(X_res)
# df_balanced[target_col] = y_res

# Method 4: RandomOverSampler (from imblearn)
# -----------------------------------------------
# sampler = RandomOverSampler(random_state=42)
# df_encoded = pd.get_dummies(df.drop(columns=[target_col]))  # Optional if you have categorical features
# X_res, y_res = sampler.fit_resample(df_encoded, df[target_col])
# df_balanced = pd.DataFrame(X_res)
# df_balanced[target_col] = y_res

# Method 5: RandomUnderSampler (from imblearn)
# -----------------------------------------------
# sampler = RandomUnderSampler(random_state=42)
# df_encoded = pd.get_dummies(df.drop(columns=[target_col]))
# X_res, y_res = sampler.fit_resample(df_encoded, df[target_col])
# df_balanced = pd.DataFrame(X_res)
# df_balanced[target_col] = y_res

# Method 6: No Balancing (use raw dataset)
# -----------------------------------------------
# df_balanced = df.sample(frac=1, random_state=42).reset_index(drop=True)

try:
    # Shuffle (if needed)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
except:
    df_balanced = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into 2 subsets
half = len(df_balanced) // 2
extra = len(df_balanced) % 2

subset_A = df_balanced.iloc[:half + extra].reset_index(drop=True)
subset_B = df_balanced.iloc[half + extra:].reset_index(drop=True)

# Print balance
print("Subset A recurrence:")
print(subset_A[target_col].value_counts())
print("\nSubset B recurrence:")
print(subset_B[target_col].value_counts())

# Save to CSV
subset_A.to_csv("BreastCancerDataRoystonAltman_subset_A.csv", index=False)
subset_B.to_csv("BreastCancerDataRoystonAltman_subset_B.csv", index=False)

print("\nSubsets saved with selected balancing method.")
