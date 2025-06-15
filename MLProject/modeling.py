# students_autolog_local.py
import warnings
import json
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, log_loss, precision_score,
    recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------
# 1. KONFIGURASI MLFLOW LOKAL
# ------------------------------------------------------------------
mlflow.set_tracking_uri("file:./mlruns")   # semua run tersimpan di ./mlruns
mlflow.set_experiment("students-experiment")   # nama eksperimen


# ------------------------------------------------------------------
# 2. UTILITAS
# ------------------------------------------------------------------
def load_dataset(path: str) -> pd.DataFrame:
    """Muat dataset dari CSV."""
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Gagal memuat dataset: {e}")


def evaluate_model(model, X_test, y_test) -> dict:
    """Hitung metrik evaluasi tambahan di test‑set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "log_loss": log_loss(y_test, y_proba),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
    }

    # ROC‑AUC multi‑kelas (atau biner)
    if len(set(y_test)) > 2:
        y_bin = label_binarize(y_test, classes=sorted(set(y_test)))
        metrics["roc_auc"] = roc_auc_score(
            y_bin, y_proba, average="weighted", multi_class="ovr"
        )
    else:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])

    return metrics


# ------------------------------------------------------------------
# 3. PIPELINE UTAMA
# ------------------------------------------------------------------
def main():
    # Aktifkan autolog sebelum training
    mlflow.sklearn.autolog(log_models=True)

    # 3‑A. Muat data
    df = load_dataset("namadataset_preprocessing/StudentsPerformance_cleaned.csv")
    label_col = df.select_dtypes(include="number").columns[-1]  # kolom target
    X = pd.get_dummies(df.drop(columns=[label_col]))
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 3‑B. Training + autolog
    with mlflow.start_run(run_name="rf_students_autolog_local"):
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # 3‑C. Tambahkan metrik test‑set
        extra_metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics({f"test_{k}": v for k, v in extra_metrics.items()})

        # 3‑D. Simpan ringkasan metrik ke artefak JSON
        summary_path = "metric_info.json"
        with open(summary_path, "w") as f:
            json.dump(extra_metrics, f, indent=2)
        mlflow.log_artifact(summary_path)

        run_id = mlflow.active_run().info.run_id
        print(f"✅ Run ID: {run_id}")
        print("📦 Artefak & metrik tersimpan di ./mlruns.")


if __name__ == "__main__":
    main()
