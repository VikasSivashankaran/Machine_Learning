import pandas as pd
import numpy as np

def count_missing_values(df):
    missing_counts = df.isnull().sum()
    return missing_counts[missing_counts > 0]

def calculate_accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = correct / total
    return accuracy

data = pd.read_csv('output.csv')

# Clean up column names (remove extra spaces)
data.columns = data.columns.str.strip()

# Convert 'Anaemic' column to binary numeric values (Yes = 1, No = 0)
data['Anaemic'] = data['Anaemic'].map({'Yes': 1, 'No': 0})


missing_counts = count_missing_values(data)
if not missing_counts.empty:
    print("DATASET - 1 \n Missing Values:")
    print(missing_counts)
else:
    print("\n DATASET - 1 \n No missing values.")

X = data.drop(['Number', 'Sex', 'Anaemic'], axis=1)  
y = data['Anaemic']

y_pred = np.zeros(len(y))

accuracy = calculate_accuracy(y, y_pred)
print("\nAccuracy:", accuracy)



def count_missing_values(df):
    missing_counts = df.isnull().sum()
    return missing_counts[missing_counts > 0]

def calculate_accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = correct / total
    return accuracy

data = pd.read_csv('missing.csv')

# Clean up column names (remove extra spaces)
data.columns = data.columns.str.strip()

# Convert 'Anaemic' column to binary numeric values (Yes = 1, No = 0)
data['Anaemic'] = data['Anaemic'].map({'Yes': 1, 'No': 0})


missing_counts = count_missing_values(data)
if not missing_counts.empty:
    print("DATASET - 2 \n Missing Values:")
    print(missing_counts)
else:
    print("No missing values.")

X = data.drop(['Number', 'Sex', 'Anaemic'], axis=1)  
y = data['Anaemic']

y_pred = np.zeros(len(y))

accuracy = calculate_accuracy(y, y_pred)
print("Accuracy:", accuracy)
