import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
data = pd.read_csv("data/student_data.csv")

print(data.to_string())

print("\nNumber of students:", len(data))
print("Average final score:", data["final_score"].mean())
print("Highest final score:", data["final_score"].max())
print("Lowest final score:", data["final_score"].min())
features = data[["study_hours", "attendance", "previous_score", "assignments"]]
target = data["final_score"]

print("\nFeatures:")
print(features)

print("\nTarget:")
print(target)
print("\nFeature shape:", features.shape)
print("Target shape:", target.shape)

X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42
)

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training targets:", y_train.shape)
print("Testing targets:", y_test.shape)

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel training completed!")

predictions = model.predict(X_test)

print("\nPredicted scores:")
print(predictions)

error = mean_absolute_error(y_test, predictions)

print("\nMean Absolute Error:", error)
print("\nActual vs Predicted:")

for actual, predicted in zip(y_test, predictions):
    print(f"Actual: {actual}, Predicted: {predicted:.2f}")

study_hours = float(input("\nEnter study hours: "))
attendance = float(input("Enter attendance percentage: "))
previous_score = float(input("Enter previous score: "))
assignments = float(input("Enter number of assignments completed: "))

new_student = pd.DataFrame(
    [[study_hours, attendance, previous_score, assignments]],
    columns=["study_hours", "attendance", "previous_score", "assignments"]
)

new_prediction = model.predict(new_student)

print("\nPredicted final score:", round(new_prediction[0], 2))
import matplotlib.pyplot as plt

plt.scatter(y_test, predictions)

plt.xlabel("Actual Final Score")
plt.ylabel("Predicted Final Score")
plt.title("Actual vs Predicted Student Scores")

plt.show()