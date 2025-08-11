# Heart Disease Prediction using Decision Tree & Random Forest

## 📌 Overview
This project predicts the presence of heart disease using tree-based machine learning models.  
We use Decision Tree and Random Forest algorithms to train, test, and compare performance on the Heart Disease dataset.

---

## 📂 Dataset
File: `heart.csv`  
Description: The dataset contains patient health metrics and a target column indicating heart disease presence (1) or absence (0).

**Key Features:**
- `age` – Age of the patient  
- `sex` – Gender (1 = male, 0 = female)  
- `cp` – Chest pain type  
- `trestbps` – Resting blood pressure  
- `chol` – Serum cholesterol level  
- `fbs` – Fasting blood sugar > 120 mg/dl  
- `restecg` – Resting electrocardiographic results  
- `thalach` – Maximum heart rate achieved  
- `exang` – Exercise induced angina  
- `oldpeak` – ST depression induced by exercise  
- `slope` – Slope of the peak exercise ST segment  
- `ca` – Number of major vessels colored by fluoroscopy  
- `thal` – Thalassemia type  
- `target` – 0 = No heart disease, 1 = Heart disease

---

## ⚙️ Requirements
Install dependencies:
```bash
pip install pandas seaborn matplotlib scikit-learn
