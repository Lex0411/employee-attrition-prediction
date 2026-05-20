# Employee Attrition Prediction

Business Intelligence application for **IS 108 Final Project** — predicts whether an employee will leave the company (**Yes** / **No**) using **KNN**, **SVM**, and **ANN**.

## Quick start (Windows)

1. Double-click **`run.bat`**
2. Open **http://127.0.0.1:5000** in your browser
3. Follow the steps: Overview → Dataset → Preprocess → Train → Results → Predict

## Manual run

```bash
cd employee-attrition-prediction
pip install -r requirements.txt
python app.py
```

## Project structure

| File | Purpose |
|------|---------|
| `app.py` | Flask web application |
| `preprocessing.py` | Load data, handle missing values, encode, scale, split |
| `models.py` | KNN, SVM, ANN training and evaluation |
| `data/sample_employee_attrition.csv` | Sample dataset (60 employees) |
| `PROJECT_DOCUMENTATION.md` | Report template for submission |

## Dataset format

Your CSV/Excel must include an **`Attrition`** column with values **Yes** or **No**, plus employee feature columns (age, department, income, overtime, etc.). See the sample file for the expected format.

## Rubric coverage

- CSV/Excel import and table preview
- Dataset information (rows, columns, missing values)
- Preprocessing (missing values, encoding, scaling, 80/20 split)
- Full predictive modeling workflow (8 steps in Overview)
- KNN, SVM, ANN trained and compared
- Accuracy, Precision, Recall, F1-score, Confusion Matrix
- Live prediction for a new employee
