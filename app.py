import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

import pickle
import os

# List of your model files
# modelnames = [
#     'Decision_tree.pkl',
#     'LogisticR.pkl',
#     'Random__forest.pkl',
#     'Support_v_m.pkl'
# ]
# for finding error
# for modelname in modelnames:
#     if not os.path.exists(modelname):
#         print(f"❌ File not found: {modelname}")
#         continue

#     if os.path.getsize(modelname) == 0:
#         print(f"❌ File is empty: {modelname}")
#         continue

#     try:
#         with open(modelname, 'rb') as f:
#             pickle.load(f)
#         print(f"✅ Loaded successfully: {modelname}")
#     except Exception as e:
#         print(f"❌ Error loading {modelname}: {e}")


st.title("Heart Disease Predictor")
tab1,tab2= st.tabs(['Predict','Model Information'])

with tab1:
    age=st.number_input("Age(years)",min_value=0,max_value=130)
    sex=st.selectbox("sex",["Male","Femail"])
    chest_pain= st.selectbox("Chest Pain Type",["Typical Angina","Atypical Angina","Non-Anginal Pain","Asymptomatic"])
    resting_bp= st.number_input("Resting Blood Pressure (mm Hg)",min_value=0,max_value=300)
    cholesterol = st.number_input("Serum Cholesterol (mm/dl)",min_value=0)
    fasting_bs=st.selectbox("Fastiog Blood Sugar",["<=120 mg/dl","> 120 mg/dl"])
    resting_ecg=st.selectbox("Restion ECG Results",["Normal","ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr=st.number_input("Maximum Heart Rate Achieved",min_value=60,max_value=202)
    exercise_angina=st.selectbox("Exercise-induced Angina",["Yes","No"])
    oldpeak=st.number_input("Oldpeak (ST Depression)",min_value=0.0,max_value=10.0)
    st_slope=st.selectbox("Slope of peak Exercise St Segment ",["Upsloping", "Flat","Downsloping"])

    sex=0 if sex=="Male" else 1
    chest_pain=["Typical Angina","Asymptomatic","Non-Anginal Pain","Atypical Angina"].index(chest_pain)
    fasting_bs=1 if fasting_bs=="> 120 mg/dl" else 0
    resting_ecg = ["Normal","ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina =1 if exercise_angina== "Yes" else 0
    st_slope=["Upsloping", "Flat","Downsloping"].index(st_slope)

    input_data= pd.DataFrame({
        'Sex':[sex],
        'ChestPainType':[chest_pain],
        'RestingECG':[resting_ecg],
        'ExerciseAngina':[exercise_angina],
        'ST_Slope':[st_slope],
        'Age':[age],
        'RestingBP':[resting_bp],
        'Cholesterol':[cholesterol],
        'FastingBS': [fasting_bs],
        'MaxHR':[max_hr],
        'Oldpeak':[oldpeak]
    })

    algonames=['Decision Trees','Logistic Regression','Support Vector Machine','Random Forest']
    modelnames=['Decision_tree.pkl','LogisticR.pkl','Support_v_m.pkl','Random__forest.pkl']

    predictions=[]
    def predict_heart_disease(data):
        predictions=[]
        for modelname in modelnames:
            with open(modelname, 'rb') as f:   
                model = pickle.load(f)
            prediction= model.predict(data)
            predictions.append(prediction)
        return predictions
    
    if st.button("Submit"):
        st.subheader('Results....')
        st.markdown('--------------------')

        result=predict_heart_disease(input_data)

        for i in range(len(result)):
            st.subheader(algonames[i])
            if result[i][0] == 0:
                st.write("No heart disease detected.")
            else:
                st.write("Heart disease detected.")
            st.markdown('-------------------')


with tab2:
    import plotly.express as px
    data={'Decision Trees': 80.97, 'Logistic Regression': 85.86,'Random Forest': 84.23,'Support Vector Machine': 84.22}
    Models=list(data.keys())
    Accuracies= list(data.values())
    df=pd.DataFrame(list(zip(Models,Accuracies)),columns=['Models','Accuracies'])
    fig=px.bar(df,y='Accuracies',x='Models')
    st.plotly_chart(fig)