### Loan Approval Prediction System

This project predicts loan eligibility using machine learning. It processes historical data to provide predictions via a Flask API and a Streamlit app, evaluating factors like income, job tenure, and credit history.

---

### Key Features

1. **Loan Eligibility Analysis**: Assesses income, employment, and credit score.
2. **Flask API**: POST endpoint for loan prediction.
3. **Streamlit App**: User-friendly interface for predictions.

---

### Setup Instructions

1. **Clone Repository**: Download files:
```bash
git clone (https://github.com/safal7360/infosys)
```

2. **Set Up Environment**: Install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Train Model**: Preprocess data and train RandomForestClassifier:
```bash
python train_model.py
```

4. **Run Flask API**: Start the API:
```bash
python app.py
```
Access it at `http://127.0.0.1:5000`.

5. **Streamlit App (Optional)**: Launch the web interface:
```bash
streamlit run app.py
```

---

### API Details

**Endpoint**: `/predict` (POST)

**Example Request**:
```json
{
  "person_age": 30,
  "person_income": 45000,
  "person_home_ownership": "OWN",
  "person_emp_length": 10,
  "loan_intent": "HOME",
  "loan_grade": "B",
  "loan_amnt": 25000,
  "loan_int_rate": 12.5,
  "loan_percent_income": 0.55,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 5
}
```
**Response**:
```json
{
  "loan_eligibility": "Not Eligible"
}
```

---

### File Overview

1. **`train_model.py`**: Prepares data and trains the model.
2. **`app.py`**: Serves the Flask API.
3. **`loan_model.pkl`**: Trained model file.
4. **`requirements.txt`**: Dependency list.

---

### Dependencies

Install with:
```bash
pip install -r requirements.txt
```
Includes Flask, Scikit-learn, pandas, numpy, Streamlit, joblib, and related libraries.

---

This guide simplifies setup and operation for loan eligibility analysis.

