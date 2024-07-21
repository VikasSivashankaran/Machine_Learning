import pandas as pd
import numpy as np
from scipy.stats import norm

# Step 1: Download and Prepare the Dataset
data = pd.read_csv('diabetics.csv')

# Step 2: Define Features and Target
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
target = 'Outcome'

# Step 3: Calculate Conditional Probability Tables (CPTs)
def calculate_cpt(data, target, features):
    cpt = {}
    for feature in features:
        cpt[feature] = {}
        for value in data[target].unique():
            feature_values = data[data[target] == value][feature]
            mean, std = feature_values.mean(), feature_values.std()
            cpt[feature][value] = (mean, std)
    return cpt

cpt = calculate_cpt(data, target, features)

# Step 4: Perform Inference
def predict_diabetes(evidence, cpt, data, target, features):
    probabilities = []
    for cls in data[target].unique():
        prob_cls = len(data[data[target] == cls]) / len(data)
        prob = prob_cls
        for feature in features:
            mean, std = cpt[feature][cls]
            value = evidence[feature]
            prob *= norm.pdf(value, loc=mean, scale=std)
        probabilities.append(prob)
    
    total_prob = sum(probabilities)
    normalized_probabilities = [prob / total_prob for prob in probabilities]

    return normalized_probabilities

# Get evidence from user input
def get_user_input():
    evidence = {}
    print("Please enter the following information:")
    for feature in features:
        while True:
            try:
                value = float(input(f"{feature}: "))
                evidence[feature] = value
                break
            except ValueError:
                print("Invalid input. Please enter a numerical value.")
    return evidence

# Example usage
evidence = get_user_input()

# Predict the probability of having diabetes
probabilities = predict_diabetes(evidence, cpt, data, target, features)
print(f"Probability of having diabetes: {probabilities[1]:.4f}")
print(f"Probability of not having diabetes: {probabilities[0]:.4f}")

# For classification (binary outcome), find the class with maximum probability
predicted_class = np.argmax(probabilities)
print(f"Predicted class: {'Diabetic' if predicted_class == 1 else 'Not Diabetic'}")
