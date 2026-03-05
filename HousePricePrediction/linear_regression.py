# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Step 2: Load Dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Training Data Loaded Successfully")
print(train.head())


# Step 3: Select Required Features
X = train[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = train['SalePrice']


# Step 4: Handle Missing Values
X = X.fillna(0)


# Step 5: Split Data into Training and Validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 6: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model Training Completed")


# Step 7: Validate Model
y_pred = model.predict(X_val)

print("Mean Squared Error:", mean_squared_error(y_val, y_pred))
print("R2 Score:", r2_score(y_val, y_pred))


# Step 8: Visualization (Actual vs Predicted Prices)
plt.figure(figsize=(8,6))

plt.scatter(y_val, y_pred)

# Perfect prediction reference line
plt.plot([y_val.min(), y_val.max()],
         [y_val.min(), y_val.max()],
         color='red', linewidth=2)

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")

plt.show()


# Step 9: Predict on Test Data
X_test = test[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
X_test = X_test.fillna(0)

predictions = model.predict(X_test)


# Step 10: Create Submission File
submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": predictions
})

submission.to_csv("submission.csv", index=False)

print("Submission File Created Successfully!")