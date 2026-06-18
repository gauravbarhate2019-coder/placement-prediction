"""
STEP 3 — DATA PREPROCESSING
Encode categories, scale features, split into train/test sets.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('data/students.csv')

print("=" * 55)
print("DATA PREPROCESSING")
print("=" * 55)

# ── Step 3a. Handle missing values ────────────────────────────────────────────
print(f"\nMissing values before: {df.isnull().sum().sum()}")
df.fillna(df.median(numeric_only=True), inplace=True)   # numeric
df.fillna(df.mode().iloc[0], inplace=True)               # categorical
print(f"Missing values after : {df.isnull().sum().sum()}")

# ── Step 3b. Encode categorical columns ───────────────────────────────────────
le_branch = LabelEncoder()
le_gender = LabelEncoder()

df['branch_enc'] = le_branch.fit_transform(df['branch'])
df['gender_enc'] = le_gender.fit_transform(df['gender'])

print(f"\nBranch encoding : {dict(zip(le_branch.classes_, le_branch.transform(le_branch.classes_)))}")
print(f"Gender encoding : {dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))}")

# ── Step 3c. Select features ──────────────────────────────────────────────────
FEATURES = ['cgpa','marks_10th','marks_12th','internships','projects',
            'certifications','backlogs','skills_score','communication',
            'branch_enc','gender_enc']
TARGET = 'placed'

X = df[FEATURES].values
y = df[TARGET].values

print(f"\nFeature matrix X : {X.shape}")
print(f"Target vector  y : {y.shape}  |  classes: {np.unique(y)}")

# ── Step 3d. Train / Test split ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size : {X_train.shape[0]}  ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"Test size  : {X_test.shape[0]}   ({X_test.shape[0]/len(X)*100:.0f}%)")
print(f"Train placed%: {y_train.mean()*100:.1f}%  |  Test placed%: {y_test.mean()*100:.1f}%")

# ── Step 3e. Scale features ───────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit on train only!
X_test_sc  = scaler.transform(X_test)        # apply same scale to test

print(f"\nBefore scaling — X_train mean: {X_train[:,0].mean():.2f}  std: {X_train[:,0].std():.2f}")
print(f"After  scaling — X_train mean: {X_train_sc[:,0].mean():.4f}  std: {X_train_sc[:,0].std():.4f}")

# ── Step 3f. Save preprocessed data ──────────────────────────────────────────
with open('models/preprocessed.pkl', 'wb') as f:
    pickle.dump({
        'X_train': X_train_sc, 'X_test': X_test_sc,
        'y_train': y_train,    'y_test': y_test,
        'scaler': scaler, 'feature_names': FEATURES,
        'le_branch': le_branch, 'le_gender': le_gender
    }, f)

print("\nPreprocessed data saved → models/preprocessed.pkl")
print("\nPREPROCESSING COMPLETE ✓")