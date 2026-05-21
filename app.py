"""
Employee Attrition Prediction — IS 108 Final Project
Flask web app: KNN, SVM, ANN comparison.
"""

import os
from pathlib import Path

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, session, url_for

from models import predict_one, train_all_models
from persist import clear_trained_bundle, load_trained_bundle, save_trained_bundle
from preprocessing import (
    TARGET_COLUMN,
    get_dataset_info,
    load_dataset,
    load_dataset_from_path,
    preprocess_data,
)

app = Flask(__name__)
app.secret_key = "attrition-bi-2026"
SAMPLE_PATH = Path(__file__).parent / "data" / "sample_employee_attrition.csv"


def get_df():
    """Get current dataset from session-backed store."""
    if "dataset_csv" not in session:
        return None
    from io import StringIO
    return pd.read_csv(StringIO(session["dataset_csv"]))


def set_df(df):
    """Store dataset CSV in session and clear any derived ML state."""
    from io import StringIO
    buf = StringIO()
    df.to_csv(buf, index=False)
    session["dataset_csv"] = buf.getvalue()
    # Invalidate derived state when dataset changes
    session.pop("preprocess", None)
    session.pop("results", None)
    clear_trained_bundle()
    app.config.pop("_ml_cache", None)
    app.config.pop("_models", None)


@app.route("/")
def overview():
    return render_template("overview.html")


@app.route("/dataset", methods=["GET", "POST"])
def dataset_page():
    """Handle dataset upload or load sample and show preview."""
    if request.method == "POST":
        action = request.form.get("action")
        try:
            if action == "upload" and "file" in request.files:
                f = request.files["file"]
                if f.filename:
                    df = load_dataset(f)
                    set_df(df)
                    flash(f"Loaded {len(df)} rows from {f.filename}.", "success")
            elif action == "sample":
                df = load_dataset_from_path(SAMPLE_PATH)
                set_df(df)
                flash("Sample employee attrition dataset loaded.", "success")
        except Exception as e:
            flash(str(e), "error")
        return redirect(url_for("dataset_page"))

    df = get_df()
    info = get_dataset_info(df) if df is not None else None
    preview = df.head(20).to_html(classes="data-table", index=False) if df is not None else None
    return render_template("dataset.html", info=info, preview=preview, has_data=df is not None)


@app.route("/preprocess", methods=["GET", "POST"])
def preprocess_page():
    """Run preprocessing and cache results for training steps."""
    df = get_df()
    if df is None:
        flash("Load a dataset first.", "error")
        return redirect(url_for("dataset_page"))

    summary = None
    if request.method == "POST" or "preprocess" in session:
        try:
            if request.method == "POST":
                data = preprocess_data(df)
                session["preprocess"] = {
                    "feature_names": data["feature_names"],
                    "class_names": data["class_names"],
                    "rows_used": data["rows_used"],
                    "train_size": data["train_size"],
                    "test_size_split": data["test_size"],
                    "encoded_columns": data["encoded_columns"],
                }
                # Cache preprocessed arrays and metadata for later training/prediction
                app.config["_ml_cache"] = data
                flash("Preprocessing complete. Ready to train models.", "success")
            summary = session.get("preprocess")
        except Exception as e:
            flash(str(e), "error")

    return render_template("preprocess.html", summary=summary, target=TARGET_COLUMN)


@app.route("/train", methods=["GET", "POST"])
def train_page():
    """Train all models using cached preprocessing and persist trained bundle."""
    df = get_df()
    if df is None:
        flash("Load a dataset first.", "error")
        return redirect(url_for("dataset_page"))
    if "preprocess" not in session:
        flash("Run preprocessing first.", "error")
        return redirect(url_for("preprocess_page"))

    if request.method == "POST":
        try:
            data = app.config.get("_ml_cache")
            if data is None:
                data = preprocess_data(df)
                app.config["_ml_cache"] = data
            # Train models and collect results + trained model objects
            results = train_all_models(
                data["X_train"], data["y_train"],
                data["X_test"], data["y_test"],
                data["class_names"],
            )
            session["results"] = {
                name: {k: v for k, v in r.items() if k != "model"}
                for name, r in results.items()
            }
            models = {name: r["model"] for name, r in results.items()}
            # Save models and preprocessing artifacts for later prediction
            app.config["_models"] = models
            app.config["_ml_cache"] = data
            save_trained_bundle(
                data["feature_names"],
                data["class_names"],
                data["label_encoders"],
                data["scaler"],
                models,
            )
            flash("All three models trained successfully.", "success")
        except Exception as e:
            flash(str(e), "error")
        return redirect(url_for("train_page"))

    trained = "results" in session
    return render_template("train.html", trained=trained)


@app.route("/results")
def results_page():
    """Display training results and highlight best-performing model."""
    if "results" not in session:
        flash("Train models first.", "error")
        return redirect(url_for("train_page"))
    results = session["results"]
    best = max(results.keys(), key=lambda m: results[m]["accuracy"])
    return render_template("results.html", results=results, best_model=best)


def _load_predict_state():
    """Load scaler, encoders, and models from memory or saved file."""
    data = app.config.get("_ml_cache")
    models = app.config.get("_models")

    if models:
        bundle = load_trained_bundle()
        if data is None and bundle:
            data = {
                "feature_names": bundle["feature_names"],
                "class_names": bundle["class_names"],
                "label_encoders": bundle["label_encoders"],
                "scaler": bundle["scaler"],
            }
            app.config["_ml_cache"] = data
        return data, models

    bundle = load_trained_bundle()
    if not bundle:
        return None, None

    data = {
        "feature_names": bundle["feature_names"],
        "class_names": bundle["class_names"],
        "label_encoders": bundle["label_encoders"],
        "scaler": bundle["scaler"],
    }
    models = bundle["models"]
    app.config["_ml_cache"] = data
    app.config["_models"] = models
    return data, models


@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    """Predict attrition for a single form row using a selected model."""
    if "results" not in session:
        flash("Train models first.", "error")
        return redirect(url_for("train_page"))

    data, models = _load_predict_state()
    prediction = None
    model_used = request.form.get("model", "KNN")

    if request.method == "POST":
        if not data or not models:
            flash("Models not loaded. Go to Train and click Train All Models again.", "error")
        else:
            try:
                import numpy as np

                features = data["feature_names"]
                row = {}
                # Convert submitted form values to numeric inputs expected by the model
                for col in features:
                    val = request.form.get(col, "").strip()
                    if col in data["label_encoders"]:
                        le = data["label_encoders"][col]
                        classes = [str(c) for c in le.classes_]
                        if val not in classes:
                            val = classes[0]
                        row[col] = le.transform([val])[0]
                    else:
                        row[col] = float(val) if val else 0.0

                # Scale features, run prediction, and format result
                X_row = np.array([[row[c] for c in features]])
                X_scaled = data["scaler"].transform(X_row)
                model = models.get(model_used) or list(models.values())[0]
                label, prob = predict_one(model, X_scaled, data["class_names"])
                prediction = {"label": label, "probability": prob, "model": model_used}
            except Exception as e:
                flash(f"Prediction failed: {e}", "error")

    feature_names = data["feature_names"] if data else []
    df = get_df()
    sample_row = df.iloc[0].to_dict() if df is not None else {}

    # Mark text vs numeric fields for the form (safe for HTML)
    field_types = {}
    if df is not None:
        from preprocessing import _is_numeric_column
        for col in feature_names:
            field_types[col] = "number" if _is_numeric_column(df[col]) else "text"

    return render_template(
        "predict.html",
        features=feature_names,
        field_types=field_types,
        sample=sample_row,
        prediction=prediction,
        models=list(session.get("results", {}).keys()),
    )


if __name__ == "__main__":
    print("\n  Employee Attrition Prediction")
    print("  Open: http://127.0.0.1:5000\n")
    # use_reloader=False avoids losing trained models in memory mid-demo
    app.run(debug=True, port=5000, use_reloader=False)
