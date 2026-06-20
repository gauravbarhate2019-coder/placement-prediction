"""
STEP 5 — DETAILED EVALUATION
Confusion matrix, ROC curve, feature importance, classification report.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
from sklearn.ensemble  import RandomForestClassifier
from sklearn.metrics   import (confusion_matrix, classification_report,
                               roc_curve, auc, ConfusionMatrixDisplay)

# ── Load data & best model ────────────────────────────────────────────────────
with open('models/preprocessed.pkl', 'rb') as f:
    data = pickle.load(f)
with open('models/best_model.pkl', 'rb') as f:
    best = pickle.load(f)

X_test   = data['X_test'];    y_test   = data['y_test']
X_train  = data['X_train'];   y_train  = data['y_train']
FEATURES = data['feature_names']
model    = best['model'];     model_name = best['name']

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ── Classification report ─────────────────────────────────────────────────────
print("=" * 55)
print(f"EVALUATION — {model_name}")
print("=" * 55)
print(classification_report(y_test, y_pred, target_names=['Not Placed', 'Placed']))

# ── Plots ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle(f'Evaluation Report — {model_name}', fontsize=14, fontweight='bold')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.4)

# 5a. Confusion matrix
ax1 = fig.add_subplot(gs[0, 0])
cm  = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Not Placed', 'Placed'])
disp.plot(ax=ax1, colorbar=False, cmap='Blues')
ax1.set_title('Confusion matrix', fontweight='bold')

# 5b. ROC curve
ax2 = fig.add_subplot(gs[0, 1])
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc     = auc(fpr, tpr)
ax2.plot(fpr, tpr, color='#2196F3', lw=2, label=f'AUC = {roc_auc:.3f}')
ax2.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5, label='Random')
ax2.fill_between(fpr, tpr, alpha=0.1, color='#2196F3')
ax2.set_xlabel('False Positive Rate'); ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC curve', fontweight='bold')
ax2.legend(); ax2.grid(alpha=0.3)

# 5c. Prediction probability distribution
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(y_prob[y_test==0], bins=20, alpha=0.6, color='#F44336', label='Not Placed')
ax3.hist(y_prob[y_test==1], bins=20, alpha=0.6, color='#4CAF50', label='Placed')
ax3.axvline(0.5, color='black', linestyle='--', lw=1.5, label='Threshold 0.5')
ax3.set_xlabel('Predicted probability'); ax3.set_ylabel('Count')
ax3.set_title('Prediction confidence', fontweight='bold')
ax3.legend(); ax3.grid(axis='y', alpha=0.3)

# 5d. Feature importance (only for tree-based models)
ax4 = fig.add_subplot(gs[1, :2])
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
else:
    # Use a RF trained on all data for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_

feat_imp = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)
names, vals = zip(*feat_imp)
colors = plt.cm.RdYlGn(np.array(vals) / max(vals))
bars = ax4.barh(names, vals, color=colors, edgecolor='white', height=0.6)
ax4.set_xlabel('Importance score')
ax4.set_title('Feature importance', fontweight='bold')
for bar, v in zip(bars, vals):
    ax4.text(v + 0.001, bar.get_y()+bar.get_height()/2,
             f'{v:.3f}', va='center', fontsize=9)
ax4.invert_yaxis()
ax4.grid(axis='x', alpha=0.3)

# 5e. Metrics summary bar
ax5 = fig.add_subplot(gs[1, 2])
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
metric_names = ['Accuracy','Precision','Recall','F1','AUC']
metric_vals  = [
    accuracy_score(y_test, y_pred),
    precision_score(y_test, y_pred),
    recall_score(y_test, y_pred),
    f1_score(y_test, y_pred),
    roc_auc_score(y_test, y_prob)
]
bar_colors = ['#2196F3','#4CAF50','#FF9800','#9C27B0','#F44336']
bars2 = ax5.bar(metric_names, metric_vals, color=bar_colors,
                alpha=0.85, edgecolor='white', width=0.6)
ax5.set_ylim(0, 1.1); ax5.set_title('Performance metrics', fontweight='bold')
ax5.set_ylabel('Score'); ax5.grid(axis='y', alpha=0.3)
for bar, v in zip(bars2, metric_vals):
    ax5.text(bar.get_x()+bar.get_width()/2, v+0.015,
             f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

plt.savefig('outputs/evaluation_report.png', dpi=150, bbox_inches='tight')
print("Evaluation report saved → outputs/evaluation_report.png")

# ── Top 3 important features ──────────────────────────────────────────────────
print(f"\nTop 3 features that predict placement:")
for i, (n, v) in enumerate(feat_imp[:3], 1):
    print(f"  {i}. {n:20s}  importance = {v:.4f}")