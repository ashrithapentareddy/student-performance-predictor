import joblib

# Load the trained model
model = joblib.load("student_performance_model.pkl")

print("Student Performance Predictor")
print("-----------------------------")
print("Model loaded successfully!")
# Get student information
study_hours = float(input("\nEnter study hours: "))

while study_hours < 0:
    print("Study hours cannot be negative.")
    study_hours = float(input("Enter study hours again: "))
attendance = float(input("Enter attendance percentage: "))

while attendance < 0 or attendance > 100:
    print("Attendance must be between 0 and 100.")
    attendance = float(input("Enter attendance percentage again: "))
previous_score = float(input("Enter previous score: "))

while previous_score < 0 or previous_score > 100:
    print("Previous score must be between 0 and 100.")
    previous_score = float(input("Enter previous score again: "))
assignments = float(input("Enter number of assignments completed: "))

while assignments < 0:
    print("Assignments cannot be negative.")
    assignments = float(input("Enter number of assignments again: "))

# Make prediction
prediction = model.predict([
    [study_hours, attendance, previous_score, assignments]
])

print(f"\nPredicted final score: {prediction[0]:.2f}")