# Step 0: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder

# Step 1: Load and explore the dataset
print("=== STEP 1: Loading and Exploring Data ===")
# Load the built-in Iris dataset
iris = load_iris()

# Create a DataFrame for better visualization and manipulation
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target  # Add target column
df['species_name'] = df['species'].apply(lambda x: iris.target_names[x])  # Add species names

print("Dataset shape:", df.shape)
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nBasic statistics:")
print(df.describe())

print("\nSpecies distribution:")
print(df['species_name'].value_counts())

# Step 2: Check for missing values
print("\n=== STEP 2: Data Preprocessing ===")
print("Missing values in each column:")
print(df.isnull().sum())

# Since this is a clean dataset, no missing values to handle

# Step 3: Prepare features and target variable
print("\n=== STEP 3: Preparing Features and Target ===")
# Features (X) - all measurements
X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]

# Target (y) - species
y = df['species']

print("Feature matrix shape:", X.shape)
print("Target vector shape:", y.shape)

# Step 4: Split the data into training and testing sets
print("\n=== STEP 4: Splitting Data into Train/Test Sets ===")
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training set size:", X_train.shape[0])
print("Testing set size:", X_test.shape[0])
print("Training set species distribution:")
print(pd.Series(y_train).value_counts())
print("Testing set species distribution:")
print(pd.Series(y_test).value_counts())

# Step 5: Create and train the Decision Tree model
print("\n=== STEP 5: Training Decision Tree Classifier ===")
# Create the model with some hyperparameters for better performance
dt_classifier = DecisionTreeClassifier(
    random_state=42,      # for reproducible results
    max_depth=3,          # prevent overfitting by limiting tree depth
    min_samples_split=5   # minimum samples required to split a node
)

# Train the model on the training data
dt_classifier.fit(X_train, y_train)

print("Model training completed!")
print(f"Model depth: {dt_classifier.get_depth()}")

# Step 6: Make predictions
print("\n=== STEP 6: Making Predictions ===")
# Predict on the test set
y_pred = dt_classifier.predict(X_test)

print("Predictions completed!")
print("First 10 actual values:", y_test[:10].values)
print("First 10 predictions:  ", y_pred[:10])

# Step 7: Evaluate the model
print("\n=== STEP 7: Model Evaluation ===")
# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

# Detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
plt.title('Confusion Matrix - Iris Species Classification')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Step 8: Feature importance analysis
print("\n=== STEP 8: Feature Importance Analysis ===")
feature_importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': dt_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance:")
print(feature_importance)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance in Decision Tree')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# Step 9: Visualize the decision tree
print("\n=== STEP 9: Decision Tree Visualization ===")
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(dt_classifier, 
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,
          rounded=True,
          fontsize=12)
plt.title('Decision Tree Visualization')
plt.show()

# Step 10: Test on individual samples
print("\n=== STEP 10: Testing on Individual Samples ===")
# Test the model on some individual samples from the test set
sample_indices = [0, 5, 10]  # indices from test set
for i in sample_indices:
    actual_species = iris.target_names[y_test.iloc[i]]
    predicted_species = iris.target_names[y_pred[i]]
    
    print(f"Sample {i}:")
    print(f"  Features: {X_test.iloc[i].values}")
    print(f"  Actual: {actual_species}")
    print(f"  Predicted: {predicted_species}")
    print(f"  Correct: {'✓' if actual_species == predicted_species else '✗'}")
    print()

print(f"✅ Model achieved {accuracy*100:.2f}% accuracy on test data")