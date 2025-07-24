import streamlit as st
import pandas as pd
import joblib
import warnings

from sklearn.ensemble import GradientBoostingClassifier


warnings.filterwarnings('ignore')


try:
    model = joblib.load('campus_placement_model')
except FileNotFoundError:
    st.error("Error: 'campus_placement_model' not found. "
             "Please ensure the model file is in the same directory as app.py "
             "and has been successfully saved from your Jupyter notebook.")
    st.stop() # Stop the app if model is not found

st.set_page_config(page_title="Campus Placement Predictor", layout="centered")

st.title("🎓 Campus Placement Predictor")
st.markdown("Enter the details below to predict your chances of placement.")


st.header("Applicant Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.radio("Gender", ["Male", "Female"])
    ssc_p = st.slider("Secondary School Percentage (SSC_P)", min_value=40.0, max_value=100.0, value=65.0, step=0.1)
    ssc_b = st.selectbox("Secondary School Board (SSC_B)", ["Central", "Others"])
    hsc_p = st.slider("Higher Secondary School Percentage (HSC_P)", min_value=30.0, max_value=100.0, value=65.0, step=0.1)
    hsc_b = st.selectbox("Higher Secondary School Board (HSC_B)", ["Central", "Others"])

with col2:
    hsc_s = st.selectbox("Higher Secondary Specialization (HSC_S)", ["Science", "Commerce", "Arts"])
    degree_p = st.slider("Degree Percentage (Degree_P)", min_value=45.0, max_value=100.0, value=65.0, step=0.1)
    degree_t = st.selectbox("Undergraduate Degree Type (Degree_T)", ["Sci&Tech", "Comm&Mgmt", "Others"])
    workex = st.radio("Work Experience (WorkEx)", ["Yes", "No"])
    etest_p = st.slider("Employability Test Percentage (Etest_P)", min_value=50.0, max_value=100.0, value=70.0, step=0.1)
    specialisation = st.selectbox("MBA Specialisation", ["Mkt&Fin", "Mkt&HR"])
    mba_p = st.slider("MBA Percentage (MBA_P)", min_value=50.0, max_value=80.0, value=60.0, step=0.1)


def preprocess_input(gender, ssc_p, ssc_b, hsc_p, hsc_b, hsc_s, degree_p, degree_t, workex, etest_p, specialisation, mba_p):
    gender_map = {"Male": 0, "Female": 1} 
    ssc_b_map = {"Central": 1, "Others": 0}
    hsc_b_map = {"Central": 1, "Others": 0}
    hsc_s_map = {"Science": 2, "Commerce": 1, "Arts": 0}
    degree_t_map = {"Sci&Tech": 2, "Comm&Mgmt": 1, "Others": 0}
    workex_map = {"Yes": 0, "No": 1} 
    specialisation_map = {"Mkt&HR": 1, "Mkt&Fin": 0}

    processed_gender = gender_map.get(gender)
    processed_ssc_b = ssc_b_map.get(ssc_b)
    processed_hsc_b = hsc_b_map.get(hsc_b)
    processed_hsc_s = hsc_s_map.get(hsc_s)
    processed_degree_t = degree_t_map.get(degree_t)
    processed_workex = workex_map.get(workex)
    processed_specialisation = specialisation_map.get(specialisation)

    input_data = pd.DataFrame({
        'gender': [processed_gender],
        'ssc_p': [ssc_p],
        'ssc_b': [processed_ssc_b],
        'hsc_p': [hsc_p],
        'hsc_b': [processed_hsc_b],
        'hsc_s': [processed_hsc_s],
        'degree_p': [degree_p],
        'degree_t': [processed_degree_t],
        'workex': [processed_workex],
        'etest_p': [etest_p],
        'specialisation': [processed_specialisation],
        'mba_p': [mba_p]
    })
    return input_data

if st.button("Predict Placement"):
    input_df = preprocess_input(gender, ssc_p, ssc_b, hsc_p, hsc_b, hsc_s, degree_p, degree_t, workex, etest_p, specialisation, mba_p)

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success(f"🎉 Congratulations! You are likely to be **Placed**.")
        st.write(f"Probability of Placement: **{probability[0][1]*100:.2f}%**")
    else:
        st.error(f"😔 It appears you might be **Not Placed**.")
        st.write(f"Probability of Not Placed: **{probability[0][0]*100:.2f}%**")

    st.markdown("---")
    st.info("Note: This prediction is based on the trained machine learning model and the input data provided.")