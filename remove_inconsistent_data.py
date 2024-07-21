import pandas as pd
file_path = 'food_coded_inconsistent_data.csv'
data = pd.read_csv(file_path)
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])
data['GPA'] = pd.to_numeric(data['GPA'], errors='coerce')
data['GPA'] = data['GPA'].fillna(data['GPA'].mean())
data['comfort_food'] = data['comfort_food'].str.strip().str.lower()
data['comfort_food_reasons'] = data['comfort_food_reasons'].str.strip().str.lower()
data['diet_current'] = data['diet_current'].str.strip().str.lower()
data['father_profession'] = data['father_profession'].str.strip().str.lower()
data['fav_cuisine'] = data['fav_cuisine'].str.strip().str.lower()
data['food_childhood'] = data['food_childhood'].str.strip().str.lower()
data = data.loc[:, ~data.columns.duplicated()]
cleaned_file_path = 'food_coded_cleaned_data.csv'
data.to_csv(cleaned_file_path, index=False)
print(f"Cleaned data saved to {cleaned_file_path}")
