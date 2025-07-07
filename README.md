
# Student Dropout Prediction App

This Flask-based web application predicts whether a student is likely to **drop out** or **graduate** based on selected academic and financial indicators.

## 🚀 Features

- Web-based user interface built with Flask and HTML/CSS
- Accepts 6 top features for dropout prediction:
  - Curricular units 2nd sem (approved)
  - Curricular units 1st sem (approved)
  - Curricular units 2nd sem (grade)
  - Curricular units 1st sem (grade)
  - Tuition fees up to date
  - Scholarship holder
- Uses a trained XGBoost model
- Model and scaler saved using joblib
- Clean interface with prediction and probability output

## 🧠 Model Details

- Algorithm: XGBoost Classifier
- Accuracy: ~90%
- Trained on filtered dataset (Graduate & Dropout only)

## 📂 Project Structure

```
├── app.py
├── scaler_top6.pkl
├── xgb_top6_model.pkl
├── requirements.txt
├── templates
│   └── index.html
└── README.md
```

## ⚙️ Installation

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## 📌 Notes

- Ensure your model file (`xgb_top6_model.pkl`) and scaler (`scaler_top6.pkl`) are in the root folder.
- The app expects 6 numerical inputs from the user.

## ✨ Author

Developed by Igwwe, Jane Adaora
