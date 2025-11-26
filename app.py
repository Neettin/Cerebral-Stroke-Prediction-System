from flask import Flask, render_template, request
import joblib
import os
import numpy as np
import pickle

app = Flask(__name__)

# Define mappings for categorical data
gender_map = {'male': 0, 'female': 1, 'others': 2}
hypertension_map = {'yes': 1, 'no': 0}
heart_disease_map = {'yes': 1, 'no': 0}
ever_married_map = {'yes': 1, 'no': 0}
work_type_map = {'private': 0, 'self-employed': 1, 'government': 2, 'student': 3, 'others': 4}
residence_type_map = {'urban': 0, 'rural': 1}
smoking_status_map = {'unknown': 0, 'never smoked': 1, 'smokes': 2, 'formerly smoked': 3, 'others': 4}

@app.route("/")
def index():
    return render_template("index_1.html")

@app.route("/result", methods=['POST'])
def result():
    gender = request.form['gender']
    age = int(request.form['age'])
    hypertension = request.form['hypertension']
    heart_disease = request.form['heart-disease']
    ever_married = request.form['marriage']
    work_type = request.form['worktype']
    residence_type = request.form['residency']
    avg_glucose_level = float(request.form['glucose'])
    bmi = float(request.form['bmi'])
    smoking_status = request.form['smoking']
    
    # Map categorical data to integers
    gender = gender_map.get(gender, -1)  # Use -1 for unknown values
    hypertension = hypertension_map.get(hypertension, -1)
    heart_disease = heart_disease_map.get(heart_disease, -1)
    ever_married = ever_married_map.get(ever_married, -1)
    work_type = work_type_map.get(work_type, -1)
    residence_type = residence_type_map.get(residence_type, -1)
    smoking_status = smoking_status_map.get(smoking_status, -1)

    # Create numpy array for the model
    x = np.array([gender, age, hypertension, heart_disease, ever_married, work_type, residence_type,
                  avg_glucose_level, bmi, smoking_status]).reshape(1, -1)

    # Load scaler and model
    scaler_path = os.path.join('model/scaler.pkl')
    model_path = os.path.join('model/dt.sav')
    
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    
    x = scaler.transform(x)

    dt = joblib.load(model_path)
    Y_pred = dt.predict(x)

    # Return appropriate result
    if Y_pred == 0:
        return render_template('index_2.html', prediction_html="No risk of stroke detected.Please continue monitoring the patient's health.")
    else:
        return render_template('index_2.html', prediction_html="Patient is at risk of stroke.Please provide recommendations for managing and reducing the risk of stroke.")

if __name__ == "__main__":
    app.run(debug=True, port=7384)
