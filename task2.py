# Step 1: Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

print("TensorFlow version:", tf.__version__)

# Step 2: Load and explore the MNIST dataset
print("\n=== STEP 1: Loading and Exploring MNIST Dataset ===")
# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Test data shape:", X_test.shape)
print("Test labels shape:", y_test.shape)

# Display class distribution
print("\nTraining set class distribution:")
unique, counts = np.unique(y_train, return_counts=True)
for class_label, count in zip(unique, counts):
    print(f"Class {class_label}: {count} samples")

# Step 3: Data preprocessing - optimized for faster training
print("\n=== STEP 2: Data Preprocessing ===")
# Normalize pixel values from 0-255 to 0-1 (faster convergence)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print("Normalized pixel value range: [{:.2f}, {:.2f}]".format(X_train.min(), X_train.max()))

# Reshape data to add channel dimension (required for CNN)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print("Reshaped training data:", X_train.shape)
print("Reshaped test data:", X_test.shape)

# Convert labels to categorical one-hot encoding
y_train_categorical = keras.utils.to_categorical(y_train, 10)
y_test_categorical = keras.utils.to_categorical(y_test, 10)

print("One-hot encoded labels shape:", y_train_categorical.shape)

# Step 4: Visualize sample images
print("\n=== STEP 3: Data Visualization ===")
def plot_sample_images(X, y, num_samples=10):
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        plt.title(f"Label: {y[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

print("Sample training images:")
plot_sample_images(X_train, y_train)

# Step 5: Build an optimized CNN model for faster training
print("\n=== STEP 4: Building Optimized CNN Model ===")
model = keras.Sequential([
    # First Convolutional Block - optimized with fewer filters initially
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),  # Faster convergence
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Second Convolutional Block - streamlined architecture
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Flatten and Dense Layers - reduced complexity for faster training
    layers.Flatten(),
    layers.Dense(128, activation='relu'),  # Reduced from larger size
    layers.BatchNormalization(),
    layers.Dropout(0.4),  # Slightly reduced dropout
    layers.Dense(10, activation='softmax')  # Output layer for 10 classes
])

# Display model architecture
model.summary()

# Step 6: Compile the model with optimized settings
print("\n=== STEP 5: Compiling Model ===")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Explicit learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model compiled with:")
print("- Optimizer: Adam (lr=0.001)")
print("- Loss: Categorical Crossentropy")
print("- Metrics: Accuracy")

# Step 7: Train the model with optimized parameters
print("\n=== STEP 6: Training Model ===")
# Use fewer epochs with efficient callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,  # Reduced patience for faster stopping
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # More aggressive learning rate reduction
        patience=2,
        min_lr=0.0001
    )
]

# Train with larger batch size for faster convergence
history = model.fit(
    X_train, y_train_categorical,
    batch_size=256,  # Increased batch size for speed
    epochs=15,       # Reduced maximum epochs
    validation_data=(X_test, y_test_categorical),
    callbacks=callbacks,
    verbose=1
)

# Step 8: Evaluate the model
print("\n=== STEP 7: Model Evaluation ===")
# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)

print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Test Loss: {test_loss:.4f}")

# Check if we achieved the goal of >95% accuracy
if test_accuracy > 0.95:
    print("🎉 SUCCESS: Model achieved >95% test accuracy!")
else:
    print("⚠️ Model did not reach 95% accuracy. Consider training longer or tuning hyperparameters.")

# Step 9: Plot training history
print("\n=== STEP 8: Training History Visualization ===")
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Step 10: Make predictions and visualize results
print("\n=== STEP 9: Predictions and Visualization ===")
# Make predictions on test set
y_pred_probs = model.predict(X_test, batch_size=512)  # Larger batch for faster prediction
y_pred = np.argmax(y_pred_probs, axis=1)

# Visualize predictions on 5 sample images as required by assignment
def visualize_predictions(X, y_true, y_pred, num_samples=5):
    plt.figure(figsize=(15, 6))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        
        # Color code: green for correct, red for incorrect
        color = 'green' if y_true[i] == y_pred[i] else 'red'
        plt.title(f"True: {y_true[i]}\nPred: {y_pred[i]}", color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

print("Visualizing the model's predictions on 5 sample images:")
visualize_predictions(X_test, y_test, y_pred)

# Step 11: Detailed performance analysis
print("\n=== STEP 10: Detailed Performance Analysis ===")
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix - MNIST Classification')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Classification Report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))

# Step 12: Analyze some misclassified examples
print("\n=== STEP 11: Misclassification Analysis ===")
misclassified_indices = np.where(y_pred != y_test)[0]

if len(misclassified_indices) > 0:
    print(f"Number of misclassified examples: {len(misclassified_indices)}")
    print(f"Error rate: {len(misclassified_indices)/len(y_test)*100:.2f}%")
    
    # Show some misclassified examples
    if len(misclassified_indices) >= 3:
        plt.figure(figsize=(12, 4))
        for i, idx in enumerate(misclassified_indices[:3]):
            plt.subplot(1, 3, i + 1)
            plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
            plt.title(f"True: {y_test[idx]}\nPred: {y_pred[idx]}", color='red')
            plt.axis('off')
        plt.suptitle('Sample Misclassified Examples', fontsize=16)
        plt.tight_layout()
        plt.show()
else:
    print("No misclassified examples! Perfect classification.")


print(f"✅ Final Test Accuracy: {test_accuracy*100:.2f}%")
print("✅ CNN model architecture, training loop, and evaluation complete!")
print("✅ Model predictions visualized on 5 sample images as required!")
