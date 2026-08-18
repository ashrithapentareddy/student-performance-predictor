import streamlit as st
import joblib

# Load trained model
model = joblib.load("student_performance_model.pkl")

# Page title
st.title("🎓 Student Performance Predictor")

st.caption(
    "A Machine Learning application for predicting student final scores"
)
st.divider()
st.write(
    "This machine learning model predicts a student's final score "
    "based on study hours, attendance, previous score, and assignments completed."
)

st.sidebar.header("About the Model")

st.sidebar.write(
    "The model uses Linear Regression to estimate the student's "
    "final academic score."
)

st.sidebar.write("Features used:")
st.sidebar.write("• Study Hours")
st.sidebar.write("• Attendance")
st.sidebar.write("• Previous Score")
st.sidebar.write("• Assignments Completed")

st.write("Enter the student's details below to predict their final score.")

st.success("Model loaded successfully! ✅")

st.divider()

st.subheader("📊 About This Project")
st.subheader("📈 Model Performance")

st.subheader("🔍 Feature Influence")

st.write(
    "These values show how strongly each feature influences the predicted score."
)

feature_data = {
    "Study Hours": 1.03,
    "Attendance": 0.20,
    "Previous Score": 0.64,
    "Assignments": 0.39
}

st.bar_chart(feature_data)

st.divider()

st.warning(
    "⚠️ Note: This model was trained on a small sample dataset "
    "of 16 students and is intended for educational purposes."
)
col1, col2 = st.columns(2)

with col1:
    st.metric("Test MAE", "0.17")

with col2:
    st.metric("Cross-validation MAE", "0.34")
st.write(
    "This project uses a Linear Regression machine learning model "
    "to predict a student's final score."
)

st.write(
    "The prediction is based on four factors:"
)

st.write(
    "• Study Hours\n"
    "• Attendance Percentage\n"
    "• Previous Score\n"
    "• Assignments Completed"
)

# Student inputs
study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=8.0
)

attendance = st.number_input(
    "Attendance Percentage",
    min_value=0.0,
    max_value=100.0,
    value=89.0
)

previous_score = st.number_input(
    "Previous Score",
    min_value=0.0,
    max_value=100.0,
    value=85.0
)

assignments = st.number_input(
    "Assignments Completed",
    min_value=0.0,
    value=10.0
)

# Prediction button
if st.button("Predict Final Score"):
    prediction = model.predict([
        [study_hours, attendance, previous_score, assignments]
    ])

    predicted_score = max(0, min(100, prediction[0]))

    st.subheader("Prediction Result")

    st.metric(
        label="Predicted Final Score",
        value=f"{predicted_score:.2f}"
    )

    if predicted_score >= 90:
        st.success("🌟 Excellent performance!")
    elif predicted_score >= 75:
        st.info("👍 Good performance! Keep working consistently.")
    elif predicted_score >= 60:
        st.warning("📚 Average performance. There is room for improvement.")
    else:
        st.error(
            "⚠️ Performance needs improvement. "
            "Consider increasing study time and attendance."
        )

    if st.button("Reset Inputs"):
        st.rerun()