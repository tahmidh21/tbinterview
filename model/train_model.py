import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    current_dir = Path(__file__).resolve().parent
    csv_path = current_dir / "Tb disease symptoms.csv"
    model_path = current_dir / "tb_model.pkl"
    features_path = current_dir / "features.json"

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find the CSV file at {csv_path}")
        print("Please ensure the CSV file is located in the 'model' directory.")
        sys.exit(1)

    df.columns = df.columns.str.strip()

    cols_to_drop = ["no", "id", "name", "date", "time"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male": 0, "Female": 1}).fillna(0).astype(int)

    symptom_cols = [
        "fever for two weeks",
        "coughing blood",
        "sputum mixed with blood",
        "night sweats",
        "chest pain",
        "back pain in certain parts",
        "shortness of breath",
        "weight loss",
        "body feels tired",
        "lumps that appear around the armpits and neck",
        "cough and phlegm continuously for two weeks to four weeks",
        "swollen lymph nodes",
        "loss of appetite"
    ]

    for col in symptom_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    df["symptom_count"] = df[symptom_cols].sum(axis=1)

    def get_risk_label(count):
        if count <= 4:
            return 0
        elif count <= 8:
            return 1
        else:
            return 2

    df["risk_label"] = df["symptom_count"].apply(get_risk_label)

    feature_cols = ["gender"] + symptom_cols

    X = df[feature_cols]
    y = df["risk_label"]

    print("Risk label distribution BEFORE training:")
    print(y.value_counts().sort_index())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy after training: {acc:.4f}")
    
    print("\nClassification Report:")
    target_names = ["Low", "Medium", "High"]
    present_classes = sorted(y_test.unique())
    present_target_names = [target_names[i] for i in present_classes]
    
    print(classification_report(y_test, y_pred, labels=present_classes, target_names=present_target_names))

    # --- 5-Fold Cross-Validation ---
    cv_model = RandomForestClassifier(n_estimators=200, random_state=42)
    cv_scores = cross_val_score(cv_model, X, y, cv=5, scoring="accuracy")
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Individual folds: {[round(s, 4) for s in cv_scores]}")

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred, labels=present_classes)
    print("\nConfusion Matrix:")
    # Print labeled header
    header = "Predicted ->  " + "  ".join(f"{n:>7}" for n in present_target_names)
    print(header)
    print("-" * len(header))
    for i, row_label in enumerate(present_target_names):
        row_vals = "  ".join(f"{cm[i][j]:>7}" for j in range(len(present_target_names)))
        print(f"{row_label:>10}  | {row_vals}")

    # Save confusion matrix as PNG
    cm_path = current_dir / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix — RandomForestClassifier", fontsize=12, pad=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(present_target_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(present_target_names)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(present_target_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    # Annotate each cell with the count
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"\nConfusion matrix saved to {cm_path}")

    # --- Baseline Comparison: Logistic Regression ---
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    print(f"\nBaseline LogisticRegression accuracy: {lr_acc:.4f} vs RandomForest accuracy: {acc:.4f}")

    joblib.dump(model, model_path)
    with open(features_path, "w") as f:
        json.dump(feature_cols, f)

    # Save feature importances
    importance_path = current_dir / "feature_importance.json"
    importances = model.feature_importances_
    feature_importances = [{"feature": feat, "importance": float(imp)} for feat, imp in zip(feature_cols, importances)]
    feature_importances.sort(key=lambda x: x["importance"], reverse=True)
    
    with open(importance_path, "w") as f:
        json.dump(feature_importances, f)

    print(f"\nModel saved to {model_path}")
    print(f"Features saved to {features_path}")
    print(f"Feature importance saved to {importance_path}")

if __name__ == "__main__":
    main()
