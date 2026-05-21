# Employee Attrition Prediction 

**Course:** IS 108 – Intelligence System Final Project SY 2025-2026

---

## Title

Employee Attrition Prediction Using KNN, SVM, and Artificial Neural Network

## Team Members

|           Name         |
|------------|-----------|
| Alexia Sheen E. Cabase | 
| Joshua Kyle S. Cabalan | 
| Ted Theone Cabanete    | 

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
