import os, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE, "data", "crop_data.csv")
MODEL_DIR  = os.path.join(BASE, "models")
REPORT_DIR = os.path.join(BASE, "reports")
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ── Load & split ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  CROP RECOMMENDATION — MODEL TRAINING PIPELINE")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET   = "label"

X = df[FEATURES].values
le = LabelEncoder()
y = le.fit_transform(df[TARGET].values)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}  |  Classes: {len(le.classes_)}\n")


# 1) Random Forest -----------------------------------------------------------
print("▶  Random Forest — GridSearchCV …")
rf_params = {
    "n_estimators": [100, 200],
    "max_depth":    [None, 10, 20],
    "min_samples_split": [2, 5],
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params, cv=5, scoring="accuracy", n_jobs=-1, verbose=0
)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
rf_acc  = accuracy_score(y_test, best_rf.predict(X_test))
rf_cv   = cross_val_score(best_rf, X_train, y_train, cv=5).mean()
results["Random Forest"] = {"model": best_rf, "test_acc": rf_acc,
                             "cv_acc": rf_cv, "params": rf_grid.best_params_,
                             "scaled": False}
print(f"   Best params : {rf_grid.best_params_}")
print(f"   CV Acc      : {rf_cv:.4f}   Test Acc: {rf_acc:.4f}\n")

# 2) SVM ---------------------------------------------------------------------
print("▶  SVM — GridSearchCV …")
svm_params = {
    "C":      [1, 10],
    "kernel": ["rbf", "poly"],
    "gamma":  ["scale", "auto"],
}
svm_grid = GridSearchCV(
    SVC(probability=True, random_state=42),
    svm_params, cv=5, scoring="accuracy", n_jobs=-1, verbose=0
)
svm_grid.fit(X_train_sc, y_train)
best_svm = svm_grid.best_estimator_
svm_acc  = accuracy_score(y_test, best_svm.predict(X_test_sc))
svm_cv   = cross_val_score(best_svm, X_train_sc, y_train, cv=5).mean()
results["SVM"] = {"model": best_svm, "test_acc": svm_acc,
                  "cv_acc": svm_cv, "params": svm_grid.best_params_,
                  "scaled": True}
print(f"   Best params : {svm_grid.best_params_}")
print(f"   CV Acc      : {svm_cv:.4f}   Test Acc: {svm_acc:.4f}\n")

# 3) KNN ---------------------------------------------------------------------
print("▶  KNN — GridSearchCV …")
knn_params = {
    "n_neighbors": [3, 5, 7, 11],
    "weights":     ["uniform", "distance"],
    "metric":      ["euclidean", "manhattan"],
}
knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    knn_params, cv=5, scoring="accuracy", n_jobs=-1, verbose=0
)
knn_grid.fit(X_train_sc, y_train)
best_knn = knn_grid.best_estimator_
knn_acc  = accuracy_score(y_test, best_knn.predict(X_test_sc))
knn_cv   = cross_val_score(best_knn, X_train_sc, y_train, cv=5).mean()
results["KNN"] = {"model": best_knn, "test_acc": knn_acc,
                  "cv_acc": knn_cv, "params": knn_grid.best_params_,
                  "scaled": True}
print(f"   Best params : {knn_grid.best_params_}")
print(f"   CV Acc      : {knn_cv:.4f}   Test Acc: {knn_acc:.4f}\n")

# ── Summary table ────────────────────────────────────────────────────────────
print("=" * 60)
print("  MODEL COMPARISON")
print("=" * 60)
summary = pd.DataFrame({
    name: {"CV Accuracy": d["cv_acc"], "Test Accuracy": d["test_acc"]}
    for name, d in results.items()
}).T
print(summary.to_string())
print()

best_name = max(results, key=lambda k: results[k]["test_acc"])
print(f"  ★  Best model: {best_name}  (Test Acc = {results[best_name]['test_acc']:.4f})")

# ── Save artefacts ───────────────────────────────────────────────────────────
joblib.dump(results[best_name]["model"], os.path.join(MODEL_DIR, "best_model.pkl"))
joblib.dump(scaler,                       os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(le,                           os.path.join(MODEL_DIR, "label_encoder.pkl"))
joblib.dump({
    "best_model_name": best_name,
    "results_summary": {k: {"test_acc": v["test_acc"],
                             "cv_acc":   v["cv_acc"],
                             "params":   v["params"]}
                        for k, v in results.items()},
    "feature_names": FEATURES,
    "classes": list(le.classes_),
}, os.path.join(MODEL_DIR, "metadata.pkl"))

print(f"\n  Artefacts saved to {MODEL_DIR}/\n")

# ── Plots ────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")

# (a) Accuracy bar chart
fig, ax = plt.subplots(figsize=(8, 4))
names  = list(results.keys())
cv_acc = [results[n]["cv_acc"]   for n in names]
te_acc = [results[n]["test_acc"] for n in names]
x = np.arange(len(names))
bars1 = ax.bar(x - 0.2, cv_acc, 0.38, label="CV Accuracy",   color="#4CAF50", alpha=0.85)
bars2 = ax.bar(x + 0.2, te_acc, 0.38, label="Test Accuracy",  color="#2196F3", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=12)
ax.set_ylim(0.8, 1.02); ax.set_ylabel("Accuracy")
ax.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
ax.legend()
for bar in [*bars1, *bars2]:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "model_comparison.png"), dpi=150)
plt.close()

# (b) Confusion matrix for best model
best = results[best_name]
X_eval = X_test_sc if best["scaled"] else X_test
y_pred = best["model"].predict(X_eval)
cm     = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=le.classes_, yticklabels=le.classes_,
            linewidths=0.5, ax=ax, annot_kws={"size": 8})
ax.set_title(f"Confusion Matrix — {best_name}", fontsize=14, fontweight="bold")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# (c) Feature importances (RF only)
rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(7, 4))
colors = ["#4CAF50" if i == importances.idxmax() else "#90CAF9" for i in importances.index]
importances.plot(kind="barh", ax=ax, color=colors)
ax.set_title("Feature Importances — Random Forest", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "feature_importance.png"), dpi=150)
plt.close()

# (d) EDA — pairplot subset
eda_df = df[FEATURES + [TARGET]].copy()
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
pairs = [("N","P"),("K","temperature"),("humidity","ph"),
         ("rainfall","N"),("temperature","humidity"),("ph","rainfall")]
sample_df = eda_df.groupby(TARGET).head(20)
for ax, (fx, fy) in zip(axes.flatten(), pairs):
    for crop in sample_df[TARGET].unique():
        sub = sample_df[sample_df[TARGET] == crop]
        ax.scatter(sub[fx], sub[fy], alpha=0.5, s=18, label=crop)
    ax.set_xlabel(fx); ax.set_ylabel(fy)
    ax.set_title(f"{fx} vs {fy}", fontsize=10)
plt.suptitle("EDA — Feature Relationships by Crop", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "eda_scatter.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"  Plots saved to {REPORT_DIR}/\n")
print("  Training pipeline complete ✔\n")
