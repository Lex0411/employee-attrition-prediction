# Employee Attrition Prediction — Project Documentation

**Course:** IS 108 – Intelligence System Final Project SY 2025-2026

---

## Title

Employee Attrition Prediction Using KNN, SVM, and Artificial Neural Network

## Team Members

| Name | Role |
|------|------|
| *(Fill in)* | |
| *(Fill in)* | |
| *(Fill in)* | |

## Business Problem Addressed

Employee turnover (attrition) is costly for organizations. This application predicts whether an employee is likely to **leave** (`Yes`) or **stay** (`No`) based on HR and job-related attributes. HR teams can use these predictions to focus retention efforts on at-risk employees.

## Dataset Used

- **Built-in sample:** `data/sample_employee_attrition.csv` (60 employee records)
- **Custom data:** Users may upload CSV or Excel files with the same structure
- **Target variable:** `Attrition` (Yes / No)
- **Features:** Age, BusinessTravel, Department, DistanceFromHome, Education, EnvironmentSatisfaction, Gender, JobRole, JobSatisfaction, MaritalStatus, MonthlyIncome, NumCompaniesWorked, OverTime, PerformanceRating, TotalWorkingYears, YearsAtCompany, WorkLifeBalance

## Data Preprocessing Steps

1. Remove rows with missing target (`Attrition`)
2. Fill missing feature values (median for numeric, mode for categorical)
3. Label-encode categorical columns
4. StandardScaler on all features
5. Train/test split (80% / 20%, stratified when possible)

## Algorithm Implementation

### K-Nearest Neighbor (KNN)
- Classifies an employee by comparing to the 5 most similar employees in training data
- Implemented with `sklearn.neighbors.KNeighborsClassifier`

### Support Vector Machine (SVM)
- Finds a decision boundary between employees who left vs. stayed
- RBF kernel, implemented with `sklearn.svm.SVC`

### Artificial Neural Network (ANN)
- Multi-layer perceptron with hidden layers (64, 32 neurons)
- Implemented with `sklearn.neural_network.MLPClassifier`

## Evaluation Metrics Used

- **Accuracy** — overall correct predictions
- **Precision** — weighted precision across classes
- **Recall** — weighted recall across classes
- **F1-Score** — harmonic mean of precision and recall
- **Confusion Matrix** — actual vs. predicted counts per class

## Comparison of Results

*(Run the application and paste your results table here after training on the sample dataset.)*

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| KNN   |          |           |        |          |
| SVM   |          |           |        |          |
| ANN   |          |           |        |          |

**Best model:** *(fill after demo)*

## Conclusion and Recommendations

- All three algorithms address the same classification problem and can be compared side-by-side in the app.
- *(Add 2–3 sentences on which model performed best on your data and why HR might use predictions alongside human judgment.)*
- Recommend retraining when new employee data is available and validating predictions before major HR decisions.

## How to Demo (10–15 min presentation)

1. Explain the business problem (employee attrition)
2. Load sample dataset → show table and dataset info
3. Run preprocessing → explain steps
4. Train all three models
5. Compare metrics and confusion matrices
6. Enter one employee profile → show prediction
7. Summarize best model and recommendations
