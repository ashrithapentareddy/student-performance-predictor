import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


print("Student Performance Predictor")
print("-----------------------------")

# 1. Load the dataset
data = pd.read_csv("data/student_data.csv")


# 2. Separate features and target
features = data[[
    "study_hours",
    "attendance",
    "previous_score",
    "assignments"
]]

target = data["final_score"]


# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42
)


# 4. Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)
joblib.dump(model, "student_performance_model.pkl")
print("\nFeature Coefficients:")

for feature, coefficient in zip(features.columns, model.coef_):
    print(f"{feature}: {coefficient:.2f}")

# 5. Test the model
predictions = model.predict(X_test)

error = mean_absolute_error(y_test, predictions)
cv_scores = cross_val_score(
    model,
    features,
    target,
    cv=4,
    scoring="neg_mean_absolute_error"
)

cv_mae = -cv_scores.mean()
print("\nModel Performance")
print("-----------------")
print("Model trained successfully!")
print("Test MAE:", round(error, 2))
print("Cross-validation MAE:", round(cv_mae, 2))


# 6. Get information about a new student
study_hours = float(input("\nEnter study hours: "))
attendance = float(input("Enter attendance percentage: "))
previous_score = float(input("Enter previous score: "))
assignments = float(input("Enter number of assignments completed: "))


# 7. Create the new student's data
new_student = pd.DataFrame(
    [[study_hours, attendance, previous_score, assignments]],
    columns=[
        "study_hours",
        "attendance",
        "previous_score",
        "assignments"
    ]
)


# 8. Predict the final score
new_prediction = model.predict(new_student)


print("\nPredicted final score:", round(new_prediction[0], 2))

import matplotlib.pyplot as plt

# 9. Visualize actual vs predicted scores
plt.figure(figsize=(7, 5))

plt.scatter(y_test, predictions, s=80)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Actual Final Score")
plt.ylabel("Predicted Final Score")
plt.title("Actual vs Predicted Student Scores")

plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()