# 🎓 Student Performance Predictor

A machine learning application that predicts a student's final score based on study habits and academic performance.

## 📌 Project Overview

The Student Performance Predictor uses Linear Regression to predict a student's final score using four input features:

- Study hours
- Attendance percentage
- Previous score
- Number of assignments completed

The trained machine learning model is integrated into an interactive Streamlit web application.

## ✨ Features

- 📊 Student performance dataset analysis
- 🧠 Linear Regression machine learning model
- 📈 Mean Absolute Error evaluation
- 🔄 Cross-validation
- 🔍 Feature coefficient visualization
- 🎯 Interactive score prediction
- 🌐 Streamlit web interface
- 💾 Saved trained model using Joblib

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit
- Git & GitHub

## 📂 Project Structure

```text
student-performance-predictor/
├── app.py
├── main.py
├── main_clean.py
├── predict.py
├── student_performance_model.pkl
├── README.md
├── .gitignore
└── data/
    └── student_data.csv

## 🧠 Machine Learning Model

The project uses **Linear Regression** to predict a student's final score based on four input features:

- Study Hours
- Attendance Percentage
- Previous Score
- Number of Assignments Completed

The dataset is divided into training and testing sets using an **80:20 split**.

The model is trained using the training data and evaluated on unseen testing data using **Mean Absolute Error (MAE)**.

The trained model is saved as a `.pkl` file using **Joblib** and is later loaded by the prediction application.

Student Dataset
      ↓
Data Loading
      ↓
Feature & Target Separation
      ↓
Train/Test Split
      ↓
Linear Regression
      ↓
Model Evaluation
      ↓
Model Saving
      ↓
Streamlit Prediction App

Test MAE: 0.17
Cross-validation MAE: 0.34

| Feature        | Coefficient |
| -------------- | ----------: |
| Study Hours    |        1.03 |
| Previous Score |        0.64 |
| Assignments    |        0.39 |
| Attendance     |        0.20 |

Study Hours: 8
Attendance: 89%
Previous Score: 85
Assignments Completed: 10
Predicted Final Score: 89.87

## 🌐 Streamlit Application

The trained machine learning model is integrated into an interactive **Streamlit web application**.

Users can enter:

- Study hours
- Attendance percentage
- Previous score
- Number of assignments completed

The application then uses the trained Linear Regression model to predict the student's final score.

The application also displays:

- 📈 Model performance
- 🔍 Feature influence
- 🎯 Predicted final score
- ⚠️ Dataset limitations

The Streamlit interface provides a simple and user-friendly way to interact with the machine learning model without running the prediction code manually.

---

## 📂 Project Structure

```text
student-performance-predictor/
│
├── app.py
├── main.py
├── main_clean.py
├── predict.py
├── student_performance_model.pkl
├── README.md
├── .gitignore
│
└── data/
    └── student_data.csv

streamlit run app.py

⚠️ Limitations

This is an educational machine learning project trained on a small sample dataset of 16 students.

The model should not be considered a production-ready system for making real academic decisions.

A larger and more diverse dataset would be required for more reliable predictions.

🔮 Future Improvements
Use a larger and more diverse dataset
Compare multiple machine learning algorithms
Perform hyperparameter tuning
Add more student-related features
Deploy the application online
Add prediction history and analytics
👩‍💻 Author

Ashritha Pentareddy

Built as a machine learning project using Python and Streamlit.