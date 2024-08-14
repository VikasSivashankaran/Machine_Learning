import csv
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Reading the data manually from CSV
data = []
with open('apples_and_oranges.csv') as file:
    reader = csv.reader(file)
    next(reader)  # skip header if present
    for row in reader:
        data.append([float(row[0]), float(row[1]), row[2]])

# Convert data to numpy array for easier manipulation
data = np.array(data)

# Step 2: Manually splitting the data into training and testing sets
np.random.seed(1)
np.random.shuffle(data)

split_ratio = 0.8
train_size = int(split_ratio * len(data))
training_set = data[:train_size]
test_set = data[train_size:]

x_train = training_set[:, 0:2].astype(float)
y_train = training_set[:, 2]
x_test = test_set[:, 0:2].astype(float)
y_test = test_set[:, 2]

# Step 3: Manually encoding the labels
unique_labels = np.unique(y_train)
print("Unique labels in training data:", unique_labels)

# Adjust the label_map to match the actual labels
label_map = {'apple': 0, 'orange': 1}

# Ensure all labels are lowercase to match the label_map
y_train_encoded = np.array([label_map[label.lower()] for label in y_train])
y_test_encoded = np.array([label_map[label.lower()] for label in y_test])

# Step 4: SVM classifier (a very simple version for demonstration)
class SimpleSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y == 1, 1, -1)  # Convert labels to +1 or -1

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

# Step 5: Train the SVM model
classifier = SimpleSVM()
classifier.fit(x_train, y_train_encoded)

# Step 6: Visualizing the decision boundary for training and testing data
def plot_decision_boundary(X, y, model):
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01),
                           np.arange(x2_min, x2_max, 0.01))
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    
    predictions = model.predict(grid).reshape(xx1.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx1, xx2, predictions, alpha=0.75, cmap='gray')
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Convert the dictionary keys to a list for indexing
    labels_list = list(label_map.keys())

    for idx, label in enumerate(np.unique(y)):
        plt.scatter(X[y == label, 0], X[y == label, 1], 
                    label=labels_list[label], edgecolor='black')

    plt.xlabel('Weight In Grams')
    plt.ylabel('Size In cms')
    plt.legend()
    plt.title('Decision Boundary')
    plt.show()

# Plotting the training data decision boundary
plot_decision_boundary(x_train, y_train_encoded, classifier)

# Plotting the testing data decision boundary
plot_decision_boundary(x_test, y_test_encoded, classifier)

# Step 7: Calculate accuracy

def calculate_accuracy(X, y_true, model):
    y_pred = model.predict(X)
    # Convert [-1, 1] back to [0, 1]
    y_pred = np.where(y_pred == 1, 1, 0)
    
    correct_predictions = np.sum(y_pred == y_true)
    accuracy = correct_predictions / len(y_true)
    return accuracy

# Calculate accuracy for training data
train_accuracy = calculate_accuracy(x_train, y_train_encoded, classifier)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

# Calculate accuracy for testing data
test_accuracy = calculate_accuracy(x_test, y_test_encoded, classifier)
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
