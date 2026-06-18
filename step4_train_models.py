"""
STEP 4 — MODEL TRAINING & COMPARISON
Train 5 different classifiers and compare their performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score,
                                     roc_auc_score)

# ── Load preprocessed data ────────────────────────────────────────────────────
with open('models/preprocessed.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train'];  X_test = data['X_test']
y_train = data['y_train'];  y_test = data['y_test']
FEATURES = data['feature_names']

# ── Define models ─────────────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree'      : DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM'                : SVC(kernel='rbf', probability=True, random_state=42),
    'KNN'                : KNeighborsClassifier(n_neighbors=7)
}

# ── Train & evaluate ──────────────────────────────────────────────────────────
print("=" * 65)
print(f"{'Model':<22}  {'Acc':>6}  {'Prec':>6}  {'Recall':>7}  {'F1':>6}  {'AUC':>6}")
print("=" * 65)

results = {}
trained = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy'  : accuracy_score(y_test, y_pred),
        'precision' : precision_score(y_test, y_pred),
        'recall'    : recall_score(y_test, y_pred),
        'f1'        : f1_score(y_test, y_pred),
        'auc'       : roc_auc_score(y_test, y_prob)
    }
    results[name] = metrics
    trained[name] = model

    print(f"{name:<22}  {metrics['accuracy']:.4f}  "
          f"{metrics['precision']:.4f}  {metrics['recall']:.4f}  "
          f"{metrics['f1']:.4f}  {metrics['auc']:.4f}")

print("=" * 65)

# ── Find best model ───────────────────────────────────────────────────────────
best_name = max(results, key=lambda m: results[m]['f1'])
best_model = trained[best_name]
print(f"\nBest model (by F1): {best_name}  →  F1 = {results[best_name]['f1']:.4f}")

# ── Save best model ───────────────────────────────────────────────────────────
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump({'model': best_model, 'name': best_name, 'metrics': results[best_name]}, f)

# ── Plot model comparison ─────────────────────────────────────────────────────
metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'auc']
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Model Comparison', fontsize=14, fontweight='bold')

# Bar chart of all metrics
model_names = list(results.keys())
short_names  = ['LR', 'DT', 'RF', 'SVM', 'KNN']
x = np.arange(len(model_names))
width = 0.15
palette = ['#2196F3','#4CAF50','#FF9800','#9C27B0','#F44336']

for i, metric in enumerate(metrics_list):
    vals = [results[m][metric] for m in model_names]
    axes[0].bar(x + i * width, vals, width, label=metric.capitalize(),
                color=palette[i], alpha=0.85, edgecolor='white')

axes[0].set_xticks(x + width * 2)
axes[0].set_xticklabels(short_names)
axes[0].set_ylim(0, 1.12)
axes[0].set_ylabel('Score')
axes[0].set_title('All metrics per model')
axes[0].legend(loc='lower right', fontsize=9)
axes[0].grid(axis='y', alpha=0.3)

# F1 ranking
f1_vals = [results[m]['f1'] for m in model_names]
bar_colors = ['#4CAF50' if m == best_name else '#90CAF9' for m in model_names]
axes[1].barh(short_names, f1_vals, color=bar_colors, edgecolor='white', height=0.5)
axes[1].set_xlim(0, 1)
axes[1].set_title('F1 score ranking')
axes[1].set_xlabel('F1 score')
for i, v in enumerate(f1_vals):
    axes[1].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)
axes[1].axvline(max(f1_vals), color='green', linestyle='--', alpha=0.5)
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/model_comparison.png', dpi=150, bbox_inches='tight')
print("Model comparison chart saved → outputs/model_comparison.png")

# ── Save all results to CSV ───────────────────────────────────────────────────
pd.DataFrame(results).T.to_csv('outputs/model_results.csv')
print("Results table saved → outputs/model_results.csv")