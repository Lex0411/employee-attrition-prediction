"""KNN, SVM, and ANN models for employee attrition prediction."""

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def train_knn(X_train, y_train, n_neighbors=5):
    """Train K-Nearest Neighbors classifier and return fitted model."""
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train):
    """Train RBF-kernel SVM with probability estimates enabled."""
    model = SVC(kernel="rbf", probability=True, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_ann(X_train, y_train):
    """Train a small MLP neural network with early stopping and return it."""
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=42,
        early_stopping=True,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, class_names):
    """Return metrics and confusion matrix for one model."""
    y_pred = model.predict(X_test)

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "class_names": class_names,
        "predictions_sample": [class_names[p] for p in y_pred[:10].tolist()],
        "actual_sample": [class_names[a] for a in y_test[:10].tolist()],
    }


def train_all_models(X_train, y_train, X_test, y_test, class_names):
    """Train KNN, SVM, and ANN; evaluate and return a results dict.

    Each entry contains metrics and the fitted `model` object.
    """
    results = {}

    knn = train_knn(X_train, y_train)
    results["KNN"] = {"model": knn, **evaluate_model(knn, X_test, y_test, class_names)}

    svm = train_svm(X_train, y_train)
    results["SVM"] = {"model": svm, **evaluate_model(svm, X_test, y_test, class_names)}

    ann = train_ann(X_train, y_train)
    results["ANN"] = {"model": ann, **evaluate_model(ann, X_test, y_test, class_names)}

    return results


def predict_one(model, features_scaled, class_names):
    """Predict attrition label and optional probability for one sample.

    Returns `(label, prob)` where `prob` is percentage or `None`.
    """
    pred_idx = model.predict(features_scaled)[0]
    label = class_names[pred_idx]
    prob = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features_scaled)[0]
        prob = round(float(max(probs)) * 100, 1)
    return label, prob
