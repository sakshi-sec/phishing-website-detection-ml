"""
ML-Based Phishing Website Detection System - model training script
Author : Sakshi Wagh (Roll No. 67)

Trains six classifiers on the Phishing Websites dataset, compares them,
and saves the best model (Random Forest) together with the evaluation results.
Run:  python train_model.py
"""
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

RANDOM_STATE = 42

# 1. Load and clean the dataset ------------------------------------------------
df = pd.read_csv("phisingData.csv")
print("Raw shape        :", df.shape)
print("Missing values   :", int(df.isnull().sum().sum()))
print("Duplicate rows   :", int(df.duplicated().sum()))
df = df.drop_duplicates().reset_index(drop=True)      # prevents train/test leakage
print("Shape after dedup:", df.shape)

X = df.drop(columns="Result")
y = (df["Result"] == -1).astype(int)                  # 1 = phishing, 0 = legitimate

# 2. Train-test split (80:20, stratified) --------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

# 3. Candidate models ----------------------------------------------------------
models = {
    "Logistic Regression": Pipeline([("scale", StandardScaler()),
                                     ("clf", LogisticRegression(max_iter=1000))]),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "K-Nearest Neighbours": Pipeline([("scale", StandardScaler()),
                                      ("clf", KNeighborsClassifier(n_neighbors=5))]),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": Pipeline([("scale", StandardScaler()),
                                        ("clf", SVC(kernel="rbf", probability=True,
                                                    random_state=RANDOM_STATE))]),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                            n_jobs=-1),
}

# 4. Train and evaluate --------------------------------------------------------
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    results[name] = {
        "acc": accuracy_score(y_test, pred),
        "prec": precision_score(y_test, pred),
        "rec": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "auc": roc_auc_score(y_test, prob),
        "cm": confusion_matrix(y_test, pred).tolist(),
    }
    print(f"{name:24s} acc={results[name]['acc']:.4f}  f1={results[name]['f1']:.4f}")

best_name = max(results, key=lambda k: results[k]["f1"])
best_model = models[best_name]
print("Best model:", best_name)

# 5. Cross-validation of the best model, feature importance --------------------
cv = cross_val_score(best_model, X, y, scoring="accuracy",
                     cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE))
importance = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)

# 6. Save artefacts ------------------------------------------------------------
joblib.dump({"model": best_model, "features": list(X.columns)}, "phishing_model.joblib")
with open("model_results.json", "w") as f:
    json.dump({"res": results, "best": best_name, "n_total": len(df),
               "n_train": len(X_train), "n_test": len(X_test),
               "n_phish": int(y.sum()), "n_legit": int((1 - y).sum()),
               "cv_mean": float(cv.mean()), "cv_std": float(cv.std()),
               "imp": importance.to_dict()}, f, indent=2)
print(f"5-fold CV accuracy: {cv.mean():.4f} +/- {cv.std():.4f}")
print("Saved phishing_model.joblib and model_results.json")
