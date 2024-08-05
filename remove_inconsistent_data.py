import pandas as pd

# Load the original data
file_path = 'food_coded_inconsistent_data.csv'
original_data = pd.read_csv(file_path)

# Save a copy of the original data before cleaning
data_before_cleaning = original_data.copy()

# Calculate the percentage of non-missing values in the original data
non_missing_before = original_data.notnull().mean() * 100

# Fill numerical columns with their mean values
numerical_cols = original_data.select_dtypes(include=['float64', 'int64']).columns
original_data[numerical_cols] = original_data[numerical_cols].fillna(original_data[numerical_cols].mean())

# Fill categorical columns with their mode values
categorical_cols = original_data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    original_data[col] = original_data[col].fillna(original_data[col].mode()[0])

# Convert GPA to numeric and handle errors
original_data['GPA'] = pd.to_numeric(original_data['GPA'], errors='coerce')
original_data['GPA'] = original_data['GPA'].fillna(original_data['GPA'].mean())

# Clean string columns
string_columns = ['comfort_food', 'comfort_food_reasons', 'diet_current', 'father_profession', 'fav_cuisine', 'food_childhood']
for col in string_columns:
    original_data[col] = original_data[col].str.strip().str.lower()

# Remove duplicated columns
original_data = original_data.loc[:, ~original_data.columns.duplicated()]

# Calculate the percentage of non-missing values in the cleaned data
non_missing_after = original_data.notnull().mean() * 100

# Save the cleaned data
cleaned_file_path = 'food_coded_cleaned_data.csv'
original_data.to_csv(cleaned_file_path, index=False)
print(f"Cleaned data saved to {cleaned_file_path}")

# Calculate overall accuracy for both datasets
accuracy_before = non_missing_before.mean()
accuracy_after = non_missing_after.mean()

print(f"Accuracy before cleaning: {accuracy_before:.2f}%")
print(f"Accuracy after cleaning: {accuracy_after:.2f}%")

# Calculate mean and standard deviation for numerical columns before and after cleaning
means_before_cleaning = data_before_cleaning[numerical_cols].mean()
std_devs_before_cleaning = data_before_cleaning[numerical_cols].std()

means_after_cleaning = original_data[numerical_cols].mean()
std_devs_after_cleaning = original_data[numerical_cols].std()

print("Means before cleaning:")
print(means_before_cleaning)
print("\nStandard deviations before cleaning:")
print(std_devs_before_cleaning)

print("\nMeans after cleaning:")
print(means_after_cleaning)
print("\nStandard deviations after cleaning:")
print(std_devs_after_cleaning)
