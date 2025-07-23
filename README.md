# Cerebral-Stroke-Prediction-System
This project predicts the likelihood of a patient having a cerebral stroke using machine learning. It uses an imbalanced healthcare dataset sourced from Kaggle and provides a simple web interface for predictions.

# 📊 Dataset
Source: Cerebral Stroke Prediction (Imbalanced Dataset) - Kaggle
Target Variable: stroke → 1 (stroke) / 0 (no stroke)
Features:
Age, Hypertension, Heart Disease, BMI, Avg Glucose Level, Smoking Status, etc.
Since the dataset is imbalanced (very few stroke cases), special handling was done during training.

# 🏥 Problem Statement
Type: Binary Classification
Goal: Predict if a patient is at risk of stroke
Challenge: Highly imbalanced dataset.

# ✅ Solution
1. Data Preprocessing: Handle missing values, encode categorical features, scale numerical features
2. Model Training: Train ML models (Logistic Regression, Random Forest, etc.)
3. Model Selection: Choose the best-performing model.
4. Deployment: Use Flask to deploy a simple web app where users can input patient details and get stroke predictions.

# Project Structure
stroke/
│── app.py                # Flask web application  
│── Cerebral_Stroke.ipynb # Jupyter Notebook for analysis & training  
│── dataset.csv           # Original dataset from Kaggle  
│── sample_data.csv       # Sample test data  
│── model/                # Saved trained model  
│── templates/            # HTML templates for web UI  
│── requirements.txt      # Python dependencies  
│── README.md             # Project documentation  

# 🚀 Installation & Setup
git clone <your-repo-url>
cd stroke

# Install Dependencies
pip install -r requirements.txt

# ▶️ Running the Application
1. Open the project in VS Code.
2. Open a terminal in the project folder.
3. Run:
   python app.py
   
# 💡 How It Works
1. The trained ML model is loaded when the Flask app starts.
2. User enters patient details in the form.
3. The model predicts stroke risk (Yes/No).
4. The result is displayed on the web page.

# Conclusion
The Cerebral Stroke Prediction System demonstrates how machine learning can assist in early detection of stroke risks, potentially helping doctors and patients take preventive actions. By using an imbalanced healthcare dataset, the system applies proper preprocessing, model training, and deployment through a Flask web app.
This project highlights the importance of data-driven decision-making in healthcare and can be further improved with larger datasets, better balancing techniques, and more advanced models for higher accuracy.

