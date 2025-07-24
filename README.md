# Campus-Placement-Predictor
This project implements a machine learning model to predict whether a student will be placed in a campus recruitment drive based on various academic and personal factors. The predictor is exposed via an interactive web application built with Streamlit.

✨ Features
Data Preprocessing: Handles categorical data encoding and prepares features for model training.

Multiple Model Evaluation: Explores several classification algorithms (Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Decision Tree, Random Forest, Gradient Boosting) to identify the best performer.

Gradient Boosting Classifier: Utilizes a Gradient Boosting Classifier as the final model due to its high accuracy.

Interactive Web GUI: A user-friendly interface built with Streamlit allows users to input their details and get instant placement predictions along with probabilities.

Model Persistence: The trained model is saved using joblib for easy loading and deployment.

📊 Dataset
The project uses the Placement.csv dataset, which contains information about students' academic performance and other relevant attributes during a campus placement drive.

Key Columns in the Dataset:

sl_no: Serial Number (dropped)

gender: Gender of the student (0/1 after encoding)

ssc_p: Secondary School Certificate (10th) percentage

ssc_b: Board of Secondary Education (Central/Others - 0/1 after encoding)

hsc_p: Higher Secondary Certificate (12th) percentage

hsc_b: Board of Higher Secondary Education (Central/Others - 0/1 after encoding)

hsc_s: Specialization in Higher Secondary (Science/Commerce/Arts - 0/1/2 after encoding)

degree_p: Degree percentage

degree_t: Type of Degree (Sci&Tech/Comm&Mgmt/Others - 0/1/2 after encoding)

workex: Work Experience (Yes/No - 0/1 after encoding)

etest_p: Employability Test percentage

specialisation: MBA Specialization (Mkt&HR/Mkt&Fin - 0/1 after encoding)

mba_p: MBA percentage

status: Placement Status (Placed/Not Placed - 0/1 after encoding, target variable)

salary: Salary offered (dropped, as it contains nulls for 'Not Placed' and is not a feature for prediction)

🛠️ Technologies Used
Python

Pandas: For data manipulation and analysis.

Scikit-learn: For machine learning model training and evaluation.

Streamlit: For building the interactive web application.

Joblib: For saving and loading the machine learning model.

Seaborn/Matplotlib (Optional): For data visualization (used in notebook, not directly in app.py).
