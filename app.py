import gradio as gr
import joblib
import pandas as pd
import numpy as np


try:
    scaler = joblib.load("scaler.pkl")
    kmeans = joblib.load("kmeans_model.pkl")
except FileNotFoundError:
    print("Error: 'scaler.pkl' or 'kmeans_model.pkl' not found.")
    print("Please ensure the model and scaler files are in the same directory as app.py.")
    scaler, kmeans = None, None
except Exception as e:
    print(f"An error occurred while loading the model files: {e}")
    print("This might be due to a version mismatch in the 'scikit-learn' library.")
    print("Please ensure the version used to run this app matches the version used to create the .pkl files.")
    scaler, kmeans = None, None


cluster_profiles = {
    0: "Cluster 0: Lower-Risk Profile",
    1: "Cluster 1: Higher-Risk Profile"
}

cluster_descriptions = {
    0: "Patients in this group generally exhibit metrics that suggest a lower immediate risk. Vitals like blood pressure, cholesterol, and glucose levels tend to be on the lower end compared to the other cluster.",
    1: "Patients in this group show metrics that may indicate a higher risk of health complications. This profile is often characterized by higher blood pressure, cholesterol, plasma glucose, and BMI."
}



def predict_cluster(
    age, gender, chest_pain_type, blood_pressure, cholesterol,
    max_heart_rate, exercise_angina, plasma_glucose, skin_thickness,
    insulin, bmi, diabetes_pedigree, hypertension, heart_disease
):
    if scaler is None or kmeans is None:
        return "Models not loaded. Please check server logs.", ""

    
    gender_numeric = 1 if gender == "Male" else 0
    exercise_angina_numeric = 1 if exercise_angina == "Yes" else 0
    hypertension_numeric = 1 if hypertension == "Yes" else 0
    heart_disease_numeric = 1 if heart_disease == "Yes" else 0

    
    feature_names = [
        'age', 'gender', 'chest_pain_type', 'blood_pressure', 'cholesterol',
        'max_heart_rate', 'exercise_angina', 'plasma_glucose', 'skin_thickness',
        'insulin', 'bmi', 'diabetes_pedigree', 'hypertension', 'heart_disease'
    ]

    
    input_values = [
        age, gender_numeric, chest_pain_type, blood_pressure, cholesterol,
        max_heart_rate, exercise_angina_numeric, plasma_glucose, skin_thickness,
        insulin, bmi, diabetes_pedigree, hypertension_numeric, heart_disease_numeric
    ]

    
    input_data = pd.DataFrame([input_values], columns=feature_names)

    
    scaled_data = scaler.transform(input_data)

    
    prediction = kmeans.predict(scaled_data)
    cluster_id = prediction[0]

    
    profile = cluster_profiles.get(cluster_id, "Unknown Cluster")
    description = cluster_descriptions.get(cluster_id, "No description available.")

    return profile, description



with gr.Blocks(theme=gr.themes.Soft(), css=".gr-box {border-color: #808080;}") as demo:
    gr.Markdown(
        """
        # 🩺 Patient Health Profile Clustering
        Enter the patient's metrics below to determine which health profile cluster they belong to.
        This tool uses a K-Means clustering model to segment patients into one of two groups based on their data.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Vitals & Demographics")
            age = gr.Slider(label="Age", minimum=1, maximum=100, step=1, value=55)
            gender = gr.Radio(label="Gender", choices=["Female", "Male"], value="Male")
            chest_pain_type = gr.Slider(label="Chest Pain Type", minimum=1, maximum=4, step=1, value=2)
            blood_pressure = gr.Slider(label="Resting Blood Pressure (mm Hg)", minimum=80, maximum=300, step=1, value=120)
            cholesterol = gr.Slider(label="Cholesterol (mg/dl)", minimum=100, maximum=600, step=1, value=200)
            max_heart_rate = gr.Slider(label="Maximum Heart Rate Achieved", minimum=60, maximum=220, step=1, value=150)
            exercise_angina = gr.Radio(label="Exercise Induced Angina", choices=["No", "Yes"], value="No")
            hypertension = gr.Radio(label="Hypertension", choices=["No", "Yes"], value="No")
            heart_disease = gr.Radio(label="Heart Disease", choices=["No", "Yes"], value="No")

        with gr.Column(scale=2):
            gr.Markdown("### Metabolic & Diabetes Indicators")
            plasma_glucose = gr.Slider(label="Plasma Glucose", minimum=50, maximum=300, step=1, value=100)
            skin_thickness = gr.Slider(label="Triceps Skin Fold Thickness (mm)", minimum=0, maximum=100, step=1, value=20)
            insulin = gr.Slider(label="2-Hour Serum Insulin (mu U/ml)", minimum=0, maximum=900, step=1, value=80)
            bmi = gr.Slider(label="Body Mass Index (BMI)", minimum=10, maximum=70, step=0.1, value=25.0)
            diabetes_pedigree = gr.Slider(label="Diabetes Pedigree Function", minimum=0.0, maximum=3.0, step=0.01, value=0.5)

    with gr.Row():
        predict_btn = gr.Button("Determine Patient Cluster", variant="primary")

    with gr.Row():
        with gr.Column(scale=1):
            output_profile = gr.Textbox(label="Predicted Patient Profile", interactive=False)
        with gr.Column(scale=2):
            output_description = gr.Textbox(label="Profile Description", interactive=False)


    
    predict_btn.click(
        fn=predict_cluster,
        inputs=[
            age, gender, chest_pain_type, blood_pressure, cholesterol,
            max_heart_rate, exercise_angina, plasma_glucose, skin_thickness,
            insulin, bmi, diabetes_pedigree, hypertension, heart_disease
        ],
        outputs=[output_profile, output_description]
    )


demo.launch()