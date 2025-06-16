# modeling.py
import warnings
import json
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, log_loss, precision_score,
    recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# 1. Konfigurasi MLflow lokal
# ------------------------------------------------------------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("students-experiment")


# ------------------------------------------------------------------
# 2. Utilitas
# ------------------------------------------------------------------
def load_dataset(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Gagal memuat dataset: {e}")


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "log_loss": log_loss(y_test, y_proba),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
    }

    if len(set(y_test)) > 2:
        y_bin = label_binarize(y_test, classes=sorted(set(y_test)))
        metrics["roc_auc"] = roc_auc_score(
            y_bin, y_proba, average="weighted", multi_class="ovr"
        )
    else:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])

    return metrics


# ------------------------------------------------------------------
# 3. Pipeline Utama
# ------------------------------------------------------------------
def main():
    mlflow.sklearn.autolog(log_models=False)  # model disimpan manual

    df = load_dataset("namadataset_preprocessing/StudentsPerformance_cleaned.csv")
    label_col = df.select_dtypes(include="number").columns[-1]
    X = pd.get_dummies(df.drop(columns=[label_col]))
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    with mlflow.start_run(run_name="rf_with_signature_and_input_example"):
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        # SIGNATURE & INPUT EXAMPLE
        input_example = X_train.iloc[:1]
        signature = infer_signature(X_train, model.predict(X_train))

        # Log model dengan signature dan contoh input
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )

        # Tambahkan METRIK evaluasi test set
        extra_metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics({f"test_{k}": v for k, v in extra_metrics.items()})

        # Simpan ringkasan metrik ke file JSON
        with open("metric_info.json", "w") as f:
            json.dump(extra_metrics, f, indent=2)
        mlflow.log_artifact("metric_info.json")

        # Simpan SERVING INPUT sebagai file terpisah
        serving_input = input_example.to_dict(orient="records")[0]
        with open("serving_input_example.json", "w") as f:
            json.dump(serving_input, f, indent=2)
        mlflow.log_artifact("serving_input_example.json")

        print(f"✅ Run ID: {mlflow.active_run().info.run_id}")
        print("📁 Model, input example, dan artefak lainnya telah disimpan.")


if __name__ == "__main__":
    main()
