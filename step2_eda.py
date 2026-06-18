"""
STEP 2 — EXPLORATORY DATA ANALYSIS (EDA)
Understand the dataset using pandas, numpy, and matplotlib.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

df = pd.read_csv('data/students.csv')

# ── 1. Basic info ─────────────────────────────────────────────────────────────
print("=" * 55)
print("BASIC INFORMATION")
print("=" * 55)
print(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nPlacement rate: {df['placed'].mean()*100:.1f}%")
print(f"\nStatistical summary:")
print(df.describe().round(2))

# ── 2. Visualizations ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Student Placement – Exploratory Data Analysis', fontsize=16, fontweight='bold', y=0.98)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

PLACED_COLOR   = '#4CAF50'
NOTPLACED_COLOR = '#F44336'
colors          = [NOTPLACED_COLOR, PLACED_COLOR]

# 2a. Placement distribution (pie)
ax1 = fig.add_subplot(gs[0, 0])
counts = df['placed'].value_counts()
ax1.pie(counts, labels=['Not Placed', 'Placed'], colors=colors,
        autopct='%1.1f%%', startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=2))
ax1.set_title('Overall placement split', fontweight='bold')

# 2b. CGPA distribution by placement
ax2 = fig.add_subplot(gs[0, 1])
for val, label, c in [(0,'Not placed', NOTPLACED_COLOR), (1,'Placed', PLACED_COLOR)]:
    ax2.hist(df[df['placed']==val]['cgpa'], bins=20, alpha=0.65,
             color=c, label=label, edgecolor='white')
ax2.set_title('CGPA distribution by placement', fontweight='bold')
ax2.set_xlabel('CGPA'); ax2.set_ylabel('Count')
ax2.legend(); ax2.grid(axis='y', alpha=0.3)

# 2c. Internships vs placement (bar)
ax3 = fig.add_subplot(gs[0, 2])
intern_rate = df.groupby('internships')['placed'].mean() * 100
bars = ax3.bar(intern_rate.index, intern_rate.values,
               color=plt.cm.RdYlGn(intern_rate.values / 100), edgecolor='white')
ax3.set_title('Placement rate by internships', fontweight='bold')
ax3.set_xlabel('Number of internships'); ax3.set_ylabel('Placement rate (%)')
for bar, val in zip(bars, intern_rate.values):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
             f'{val:.0f}%', ha='center', fontsize=9)
ax3.grid(axis='y', alpha=0.3); ax3.set_ylim(0, 100)

# 2d. Skills score vs placement (box)
ax4 = fig.add_subplot(gs[1, 0])
data_box = [df[df['placed']==0]['skills_score'], df[df['placed']==1]['skills_score']]
bp = ax4.boxplot(data_box, patch_artist=True, labels=['Not placed', 'Placed'],
                 widths=0.5)
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax4.set_title('Skills score by placement', fontweight='bold')
ax4.set_ylabel('Skills score (1–10)'); ax4.grid(axis='y', alpha=0.3)

# 2e. Backlogs vs placement (bar)
ax5 = fig.add_subplot(gs[1, 1])
backlog_rate = df.groupby('backlogs')['placed'].mean() * 100
bars2 = ax5.bar(backlog_rate.index, backlog_rate.values,
                color='#FF7043', edgecolor='white', alpha=0.85)
ax5.set_title('Placement rate by backlogs', fontweight='bold')
ax5.set_xlabel('Number of backlogs'); ax5.set_ylabel('Placement rate (%)')
for bar, val in zip(bars2, backlog_rate.values):
    ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
             f'{val:.0f}%', ha='center', fontsize=9)
ax5.grid(axis='y', alpha=0.3); ax5.set_ylim(0, 100)

# 2f. Branch-wise placement rate (horizontal bar)
ax6 = fig.add_subplot(gs[1, 2])
branch_rate = df.groupby('branch')['placed'].mean().sort_values() * 100
branch_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(branch_rate)))
ax6.barh(branch_rate.index, branch_rate.values, color=branch_colors, edgecolor='white')
ax6.set_title('Placement rate by branch', fontweight='bold')
ax6.set_xlabel('Placement rate (%)')
ax6.axvline(50, color='red', linestyle='--', alpha=0.5, label='50% line')
ax6.legend(fontsize=8); ax6.grid(axis='x', alpha=0.3)

# 2g. Correlation heatmap (numeric only)
ax7 = fig.add_subplot(gs[2, :])
num_cols = ['cgpa','marks_10th','marks_12th','internships','projects',
            'certifications','backlogs','skills_score','communication','placed']
corr = df[num_cols].corr()
im = ax7.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
ax7.set_xticks(range(len(num_cols))); ax7.set_yticks(range(len(num_cols)))
ax7.set_xticklabels(num_cols, rotation=35, ha='right', fontsize=9)
ax7.set_yticklabels(num_cols, fontsize=9)
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        ax7.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center',
                 fontsize=7, color='black' if abs(corr.iloc[i,j]) < 0.7 else 'white')
plt.colorbar(im, ax=ax7, shrink=0.6)
ax7.set_title('Feature correlation heatmap', fontweight='bold')

plt.savefig('outputs/eda_report.png', dpi=150, bbox_inches='tight')
print("\nEDA chart saved → outputs/eda_report.png")

# ── 3. Key insights (printed) ─────────────────────────────────────────────────
print("\n" + "="*55)
print("KEY EDA INSIGHTS")
print("="*55)
print(f"Avg CGPA  — Placed: {df[df.placed==1].cgpa.mean():.2f}  Not: {df[df.placed==0].cgpa.mean():.2f}")
print(f"Avg Skills— Placed: {df[df.placed==1].skills_score.mean():.2f}  Not: {df[df.placed==0].skills_score.mean():.2f}")
print(f"Top corr with 'placed': {corr['placed'].drop('placed').abs().idxmax()}")
import os
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)