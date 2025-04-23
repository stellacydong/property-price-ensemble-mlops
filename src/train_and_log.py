import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_class_weight

# ========== Step 1: Load and Preprocess Data ==========
df = pd.read_csv("train_ml_ops.csv")
df.drop(columns=["uid"], inplace=True)
df.dropna(inplace=True)
df["hasSpa"] = df["hasSpa"].astype(str).str.lower().map({"true": 1, "false": 0}).fillna(0).astype(int)

# Encode categorical columns
label_encoders = {}
for col in ["city", "homeType", "priceRange"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Feature engineering
df['log_lotSizeSqFt'] = np.log1p(df['lotSizeSqFt'])
df['bath_per_bed'] = df['numOfBathrooms'] / (df['numOfBedrooms'] + 1)
df['age'] = 2025 - df['yearBuilt']

# Define X and y
selected_features = [
    "avgSchoolRating", "numOfBathrooms", "MedianStudentsPerTeacher",
    "numOfBedrooms", "garageSpaces", "latitude", "longitude",
    "yearBuilt", "lotSizeSqFt", "log_lotSizeSqFt", "bath_per_bed",
    "age", "city", "homeType", "hasSpa"
]
X = df[selected_features]
y = df["priceRange"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ========== Step 2: Define Models and Ensemble ==========
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = {cls: weight for cls, weight in zip(classes, weights)}

lgb_model = LGBMClassifier(
    objective='multiclass',
    num_class=len(classes),
    class_weight=class_weight_dict,
    learning_rate=0.1,
    num_leaves=31,
    n_estimators=200,
    random_state=42
)

histgb_model = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_iter=200,
    max_leaf_nodes=31,
    min_samples_leaf=50,
    random_state=42
)

xgb_model = XGBClassifier(
    learning_rate=0.1,
    max_depth=5,
    n_estimators=100,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)

ensemble_model = VotingClassifier(
    estimators=[
        ('lgb', lgb_model),
        ('histgb', histgb_model),
        ('xgb', xgb_model)
    ],
    voting='soft',
    n_jobs=-1
)

ensemble_model.fit(X_train, y_train)
y_pred = ensemble_model.predict(X_test)

# ========== Step 3: Log to MLflow ==========
mlflow.set_experiment("Property_Price_Ensemble")
os.makedirs("mlruns_artifacts", exist_ok=True)

with mlflow.start_run(run_name="VotingEnsemble_LGB_HistGB_XGB"):
    mlflow.log_param("model_type", "VotingClassifier")
    mlflow.log_param("voting", "soft")
    mlflow.log_param("lgb_learning_rate", 0.1)
    mlflow.log_param("lgb_n_estimators", 200)
    mlflow.log_param("histgb_learning_rate", 0.05)
    mlflow.log_param("histgb_max_iter", 200)
    mlflow.log_param("xgb_learning_rate", 0.1)
    mlflow.log_param("xgb_max_depth", 5)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("macro_f1", macro_f1)
    mlflow.log_metric("weighted_f1", weighted_f1)

    mlflow.sklearn.log_model(ensemble_model, artifact_path="voting_ensemble_model")

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_path = "mlruns_artifacts/classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=4)
    mlflow.log_artifact(report_path)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix: Voting Ensemble")
    cm_path = "mlruns_artifacts/confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)

    pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    pred_path = "mlruns_artifacts/predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    mlflow.log_artifact(pred_path)

    print(f"✅ MLflow run completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

