# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Create Dataset
np.random.seed(42)

data = {
    'Task_Time': np.random.randint(2, 10, 100),
    'Feedback': np.random.uniform(1, 5, 100),
    'Attendance': np.random.randint(50, 100, 100)
}

df = pd.DataFrame(data)

# Step 3: Create Target (Performance)
df['Performance'] = (
    70 
    - df['Task_Time'] * 3 
    + df['Feedback'] * 6 
    + df['Attendance'] * 0.2
)

df['Performance'] = df['Performance'].clip(0, 100)

print("\nDataset Sample:\n", df.head())

# Step 4: Define Features & Target
X = df[['Task_Time', 'Feedback', 'Attendance']]
y = df['Performance']

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Prediction
y_pred = model.predict(X_test)

# Step 8: Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MSE:", mse)
print("R2 Score:", r2)

# Step 9: Feature Importance
print("\nFeature Importance:")
for i, col in enumerate(X.columns):
    print(col, ":", model.feature_importances_[i])

# Step 10: Predict New Intern
new_intern = pd.DataFrame([[6, 4.5, 85]], columns=['Task_Time', 'Feedback', 'Attendance'])
prediction = model.predict(new_intern)
print("\nPredicted Performance for new intern:", prediction)

features = X.columns
importance = model.feature_importances_

plt.bar(features, importance)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Performance")
plt.ylabel("Predicted Performance")
plt.title("Actual vs Predicted")
plt.show()
def categorize(score):
    if score >= 80:
        return "High"
    elif score >= 60:
        return "Medium"
    else:
        return "Low"

df['Category'] = df['Performance'].apply(categorize)

def categorize(score):
    if score >= 85:
        return "High Performer"
    elif score >= 60:
        return "Average Performer"
    else:
        return "Needs Improvement"
category = categorize(prediction[0])

print("Predicted Performance:", prediction)
print("Predicted Category:", category)