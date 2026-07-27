import streamlit as st
import pickle
import pandas as pd

with open("model_columns.pkl", "rb") as pickle_in_cols:
    model_columns = pickle.load(pickle_in_cols)

with open("diabetes_model.pkl", "rb") as pickle_in_model:
    model = pickle.load(pickle_in_model)

with open("scaler.pkl", "rb") as pickle_in_scaler:
    scaler = pickle.load(pickle_in_scaler)

def predict_note(bmi, physical_activity_minutes_per_week, systolic_bp, diastolic_bp, cholesterol_total, glucose_fasting, glucose_postprandial):
    all_numerical_cols = [
        'age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week', 
        'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi', 
        'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate', 
        'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides', 
        'glucose_fasting', 'glucose_postprandial', 'insulin_level', 'hba1c', 'diabetes_risk_score'
    ]
    
    scaler_df = pd.DataFrame([{col: 0.0 for col in all_numerical_cols}])
    
    scaler_df["bmi"] = bmi
    scaler_df["physical_activity_minutes_per_week"] = physical_activity_minutes_per_week
    scaler_df["systolic_bp"] = systolic_bp
    scaler_df["diastolic_bp"] = diastolic_bp
    scaler_df["cholesterol_total"] = cholesterol_total
    scaler_df["glucose_fasting"] = glucose_fasting
    scaler_df["glucose_postprandial"] = glucose_postprandial
    
    scaler_df[all_numerical_cols] = scaler.transform(scaler_df[all_numerical_cols])
    
    input_data = {col: 0 for col in model_columns}
    
    for col in all_numerical_cols:
        if col in input_data:
            input_data[col] = scaler_df.loc[0, col]
    
    input_df = pd.DataFrame([input_data])
    
    input_df = input_df[model_columns]
    
    prediction = model.predict(input_df)
    print(prediction)
    return prediction[0]

def main():
    st.title("Diabetes prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Diabetes Authenticator ML App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    bmi = st.number_input("BMI", min_value=0.0, step=0.1, value=25.0)
    physical_activity_minutes_per_week = st.number_input("Phy Act Mins Per Week", min_value=0, step=1, value=150)
    systolic_bp = st.number_input("systolic_bp", min_value=0, step=1, value=120)
    diastolic_bp = st.number_input("diastolic_bp", min_value=0, step=1, value=80)
    cholesterol_total = st.number_input("cholesterol_total", min_value=0, step=1, value=180)
    glucose_fasting = st.number_input("glucose_fasting", min_value=0, step=1, value=90)
    glucose_postprandial = st.number_input("glucose_postprandial", min_value=0, step=1, value=120)
    
    result = ""
    if st.button("Predict"):
        result = predict_note(bmi, physical_activity_minutes_per_week, systolic_bp, diastolic_bp, cholesterol_total, glucose_fasting, glucose_postprandial)
        
        if result == 1:
            st.error("You have high risk of diabetes. Please consult a healthcare professional.")
        else:
            st.success("You have a low risk of diabetes.")

if __name__ == '__main__':
    main()