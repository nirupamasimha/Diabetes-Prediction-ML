import streamlit as st
import pickle
import pandas as pd


# Load saved files
with open("model_columns.pkl", "rb") as file:
    model_columns = pickle.load(file)

with open("random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


def predict_note(
    bmi,
    physical_activity_minutes_per_week,
    systolic_bp,
    diastolic_bp,
    cholesterol_total,
    glucose_fasting,
    glucose_postprandial,
    cardiovascular_history,
    family_history_diabetes,
    hypertension_history
):

    # Create input dictionary
    input_data = {
        "bmi": bmi,
        "physical_activity_minutes_per_week": physical_activity_minutes_per_week,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "cholesterol_total": cholesterol_total,
        "glucose_fasting": glucose_fasting,
        "glucose_postprandial": glucose_postprandial,
        "cardiovascular_history": cardiovascular_history,
        "family_history_diabetes": family_history_diabetes,
        "hypertension_history": hypertension_history
    }

    # Get the exact numerical columns used when scaler was trained
    scaler_columns = list(scaler.feature_names_in_)

    # Create dataframe with all required scaler columns
    scaler_df = pd.DataFrame(
        [{col: 0.0 for col in scaler_columns}]
    )

    # Put user values into the dataframe
    for col, value in input_data.items():
        if col in scaler_df.columns:
            scaler_df.loc[0, col] = value

    # Automatically arrange columns in the order expected by scaler
    scaler_df = scaler_df.reindex(columns=scaler_columns)

    # Scale numerical features
    scaled_data = scaler.transform(scaler_df)

    # Convert scaled data back to DataFrame
    scaled_df = pd.DataFrame(
        scaled_data,
        columns=scaler_columns
    )

    # Create final model input with all model columns
    final_input = pd.DataFrame(
        0.0,
        index=[0],
        columns=model_columns
    )

    # Copy scaled numerical values into final input
    for col in scaler_columns:
        if col in final_input.columns:
            final_input.loc[0, col] = scaled_df.loc[0, col]

    # Automatically arrange columns in model's expected order
    final_input = final_input.reindex(columns=model_columns)

    # Prediction
    prediction = model.predict(final_input)

    return prediction[0]


def main():

    st.title("Diabetes Prediction")

    st.markdown(
        """
        <div style="background-color:tomato;padding:10px">
        <h2 style="color:white;text-align:center;">
        Streamlit Diabetes Prediction ML App
        </h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        step=0.1,
        value=25.0
    )

    physical_activity_minutes_per_week = st.number_input(
        "Physical Activity Minutes Per Week",
        min_value=0,
        step=1,
        value=150
    )

    systolic_bp = st.number_input(
        "Systolic BP",
        min_value=0,
        step=1,
        value=120
    )

    diastolic_bp = st.number_input(
        "Diastolic BP",
        min_value=0,
        step=1,
        value=80
    )

    cholesterol_total = st.number_input(
        "Total Cholesterol",
        min_value=0,
        step=1,
        value=180
    )

    glucose_fasting = st.number_input(
        "Fasting Glucose",
        min_value=0,
        step=1,
        value=90
    )

    glucose_postprandial = st.number_input(
        "Postprandial Glucose",
        min_value=0,
        step=1,
        value=120
    )

    cardiovascular_history = st.selectbox(
        "Cardiovascular History",
        [0, 1]
    )

    family_history_diabetes = st.selectbox(
        "Family History of Diabetes",
        [0, 1]
    )

    hypertension_history = st.selectbox(
        "Hypertension History",
        [0, 1]
    )

    if st.button("Predict"):

        result = predict_note(
            bmi,
            physical_activity_minutes_per_week,
            systolic_bp,
            diastolic_bp,
            cholesterol_total,
            glucose_fasting,
            glucose_postprandial,
            cardiovascular_history,
            family_history_diabetes,
            hypertension_history
        )

        if result == 1:
            st.error(
                "High risk of diabetes. Please consult a healthcare professional."
            )
        else:
            st.success(
                "Low risk of diabetes."
            )


if __name__ == "__main__":
    main()