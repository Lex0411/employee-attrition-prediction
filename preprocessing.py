"""Data preprocessing for employee attrition prediction."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

TARGET_COLUMN = "Attrition"
DEFAULT_FEATURES = [
    "Age",
    "BusinessTravel",
    "Department",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "Gender",
    "JobRole",
    "JobSatisfaction",
    "MaritalStatus",
    "MonthlyIncome",
    "NumCompaniesWorked",
    "OverTime",
    "PerformanceRating",
    "TotalWorkingYears",
    "YearsAtCompany",
    "WorkLifeBalance",
]


def load_dataset(file_storage):
    """Load CSV or Excel from upload."""
    filename = (getattr(file_storage, "filename", "") or "").lower()
    if filename.endswith(".csv"):
        return pd.read_csv(file_storage)
    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_storage)
    raise ValueError("Please upload a CSV or Excel file (.csv, .xlsx, .xls).")


def load_dataset_from_path(path):
    """Load CSV or Excel from disk."""
    path_lower = str(path).lower()
    if path_lower.endswith(".csv"):
        return pd.read_csv(path)
    if path_lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    raise ValueError("Sample file must be CSV or Excel.")


def _is_numeric_column(series):
    """True if column should be treated as numeric (not categorical text)."""
    if pd.api.types.is_numeric_dtype(series):
        return True
    if pd.api.types.is_bool_dtype(series):
        return True
    # pandas StringDtype or object columns that are actually numbers
    converted = pd.to_numeric(series, errors="coerce")
    non_null = series.notna().sum()
    if non_null == 0:
        return False
    return converted.notna().sum() / non_null >= 0.9


def _fill_missing_column(series):
    """Fill missing values with median (numeric) or mode (text)."""
    if _is_numeric_column(series):
        numeric = pd.to_numeric(series, errors="coerce")
        median_val = numeric.median()
        if pd.isna(median_val):
            median_val = 0
        return numeric.fillna(median_val)
    mode = series.dropna().mode()
    fill_val = mode.iloc[0] if len(mode) > 0 else "Unknown"
    return series.fillna(fill_val).astype(str)


def get_dataset_info(df):
    """Basic dataset summary for the UI."""
    missing = df.isnull().sum()
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": {col: int(count) for col, count in missing.items() if count > 0},
        "total_missing": int(missing.sum()),
        "target_column": TARGET_COLUMN if TARGET_COLUMN in df.columns else None,
    }


def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Full preprocessing pipeline:
    - handle missing values
    - encode categorical columns
    - scale numeric features
    - train/test split
    """
    work = df.copy()

    if TARGET_COLUMN not in work.columns:
        raise ValueError(f"Dataset must include target column '{TARGET_COLUMN}' (Yes/No).")

    # Drop rows with missing target
    work = work.dropna(subset=[TARGET_COLUMN])
    work[TARGET_COLUMN] = work[TARGET_COLUMN].astype(str).str.strip()

    feature_cols = [c for c in work.columns if c != TARGET_COLUMN]
    if not feature_cols:
        raise ValueError("Dataset needs at least one feature column besides Attrition.")

    # Fill missing values (numeric vs text — avoids median on string columns)
    for col in feature_cols:
        work[col] = _fill_missing_column(work[col])

    X = work[feature_cols]
    y_raw = work[TARGET_COLUMN]

    label_encoders = {}
    X_encoded = X.copy()

    for col in X_encoded.columns:
        if _is_numeric_column(X_encoded[col]):
            X_encoded[col] = pd.to_numeric(X_encoded[col], errors="coerce").fillna(0)
        else:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            label_encoders[col] = le

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y_raw)
    class_names = list(target_encoder.classes_)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_cols,
        "class_names": class_names,
        "label_encoders": label_encoders,
        "target_encoder": target_encoder,
        "scaler": scaler,
        "rows_used": len(work),
        "train_size": len(y_train),
        "test_size": len(y_test),
        "encoded_columns": list(label_encoders.keys()),
    }
