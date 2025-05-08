# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Preprocessing & Evaluation
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE

# Load dataset
file_path = "D:\TRIAMONDS\\embeddings\\best_mathbert_finetuned_embeddings.xlsx"
data = pd.read_excel(file_path)

X = data.drop(columns=['Class'])

y = data['Class']

# Handle missing and scale
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, class_weight='balanced', random_state=42),
    "SVM": SVC(C=1, kernel='rbf', gamma='scale', probability=True, class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(learning_rate=0.05, max_depth=8, n_estimators=150, subsample=0.8, colsample_bytree=0.8, use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "Logistic Regression": LogisticRegression(C=1.0, penalty='l2', solver='liblinear', class_weight='balanced', max_iter=1000, random_state=42),
    "Naive Bayes": GaussianNB(var_smoothing=1e-9),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_split=5, class_weight='balanced', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7, metric='minkowski', p=2),
    "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', alpha=0.0001, max_iter=300, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
}

# Store results
results_cv = {}
results_precision_recall = {}
per_class_report = defaultdict(dict)

# Stratified K-Fold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for model_name, model in models.items():
    accs, f1s, precs, recs = [], [], [], []
    classwise_reports = []
    all_true = []
    all_pred = []

    for train_idx, test_idx in skf.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # SMOTE
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

        # Fit and predict
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)

        # Metrics
        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average='weighted'))
        precs.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        recs.append(recall_score(y_test, y_pred, average='weighted'))

        classwise_reports.append(classification_report(y_test, y_pred, output_dict=True, zero_division=0))

        all_true.extend(y_test)
        all_pred.extend(y_pred)

    # Aggregated scores
    results_cv[model_name] = {
        "Mean Accuracy": np.mean(accs),
        "Std Accuracy": np.std(accs),
        "Mean F1-Score": np.mean(f1s),
        "Std F1-Score": np.std(f1s),
    }

    results_precision_recall[model_name] = {
        "Mean Precision": np.mean(precs),
        "Std Precision": np.std(precs),
        "Mean Recall": np.mean(recs),
        "Std Recall": np.std(recs),
    }

    for label in y.unique():
        label = str(label)
        avg_precision = np.mean([fold[label]['precision'] for fold in classwise_reports if label in fold])
        avg_recall = np.mean([fold[label]['recall'] for fold in classwise_reports if label in fold])
        avg_f1 = np.mean([fold[label]['f1-score'] for fold in classwise_reports if label in fold])
        per_class_report[model_name][f"Class {label} Precision"] = avg_precision
        per_class_report[model_name][f"Class {label} Recall"] = avg_recall
        per_class_report[model_name][f"Class {label} F1"] = avg_f1

    # Plot Confusion Matrix
    cm = confusion_matrix(all_true, all_pred, labels=sorted(y.unique()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y.unique()))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.show()

# Final DataFrames
cv_df = pd.DataFrame(results_cv).T.sort_values(by='Mean F1-Score', ascending=False)
prec_rec_df = pd.DataFrame(results_precision_recall).T.sort_values(by='Mean Precision', ascending=False)
per_class_df = pd.DataFrame(per_class_report).T

# Print summaries
print("\nOverall CV Results:\n")
print(cv_df)
print("\nPrecision & Recall:\n")
print(prec_rec_df)
print("\nPer-Class Stratified Results:\n")
print(per_class_df)

# Save to Excel
output_path = "D://best_final_finetuned_bert_results.xlsx"
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    cv_df.to_excel(writer, sheet_name='CrossVal_Accuracy_F1')
    prec_rec_df.to_excel(writer, sheet_name='Precision_Recall')
    per_class_df.to_excel(writer, sheet_name='Per_Class_Report')

print(f"\n✅ Results saved to: {output_path}")
