import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the housing data — same dataset we've been working with!
# Pulling it directly from sklearn so we can skip local CSV loading
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data["median_house_value"] = housing.target * 100000  # sklearn version stores it in $100k units

# Separate features and target
X = data.drop("median_house_value", axis=1).values
y = data["median_house_value"].values

# Scale the features. Neural networks are sensitive to feature scale
# Luckily there's a built-in function for it!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Features per sample:", X_train.shape[1])

# ================ Building the Model =====================
# This basically what SageMaker was hiding from us!
# We're explicitly defining each layer of the neural network

model = tf.keras.Sequential([
    # Input layer — one neuron per feature
    tf.keras.layers.Input(shape=(X_train.shape[1],)),

    # Hidden layer 1 — 64 neurons, ReLU activation (stands for Rectified Linear Unit)
    # ReLU: "if the value is negative, make it 0. Otherwise keep it."
    tf.keras.layers.Dense(64, activation="relu"),

    # Hidden layer 2 — 32 neurons
    tf.keras.layers.Dense(32, activation="relu"),

    # Output layer — 1 neuron because we're predicting one value (house price)
    # No activation function = linear output, which is what we want for regression
    tf.keras.layers.Dense(1)
])

# Print a summary of the model architecture
# Great demo moment — students can see the layers and parameter counts
model.summary()

# =========== Compiling The Model ===================

# Tell TensorFlow HOW to train the model
# optimizer: how it adjusts weights (adam is the standard choice)
# loss: what we're trying to minimize (MSE for regression)
# metrics: what we want to see during training

model.compile(
    optimizer="adam", # standard - changes weights and learning rate during training
    loss="mse", # mean squared error - we know this one. How far off the predictions are
    metrics=["mae"]  # Mean Absolute Error - avg diff between predictions and actual values.
)


# ============ Training The Model =====================
# epochs: how many times to loop through the entire training dataset
# validation_split: hold out 20% of training data to track validation loss
# This is the equivalent of estimator.fit() in SageMaker!

history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_split=0.2,
    verbose=1  # shows training progress per epoch
)

# ================ Evaluating the Model =====================
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest MAE: ${test_mae:,.0f}")
print(f"On average our predictions are off by ${test_mae:,.0f}")

# ============= Make a Prediction ==================
# Grab one sample from the test set
sample = X_test[0].reshape(1, -1)
actual = y_test[0]
predicted = model.predict(sample)[0][0]

print(f"\nActual price:    ${actual:,.0f}")
print(f"Predicted price: ${predicted:,.0f}")
print(f"Difference:      ${abs(actual - predicted):,.0f}")
