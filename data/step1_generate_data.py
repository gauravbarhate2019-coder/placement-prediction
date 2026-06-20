"""
STEP 1 — GENERATE DATASET
Creates a realistic student placement dataset with 500 students.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
n = 500

# ── Features ──────────────────────────────────────────────────────────────────
cgpa           = np.round(np.random.uniform(5.0, 10.0, n), 2)
marks_10th     = np.round(np.random.uniform(50, 100, n), 1)
marks_12th     = np.round(np.random.uniform(50, 100, n), 1)
internships    = np.random.randint(0, 4, n)
projects       = np.random.randint(0, 6, n)
certifications = np.random.randint(0, 5, n)
backlogs       = np.random.randint(0, 5, n)
skills_score   = np.random.randint(1, 11, n)          # 1–10
communication  = np.random.randint(1, 11, n)          # 1–10
branch         = np.random.choice(['CSE','ECE','Mech','Civil','IT'], n)
gender         = np.random.choice(['Male','Female'], n)

# ── Target (placement) ────────────────────────────────────────────────────────
# Placement is influenced by cgpa, internships, skills, backlogs
score = (
    0.35 * (cgpa / 10) +
    0.20 * (internships / 3) +
    0.20 * (skills_score / 10) +
    0.15 * (communication / 10) +
    0.10 * (projects / 5) -
    0.15 * (backlogs / 4)
)
noise     = np.random.normal(0, 0.08, n)
prob      = np.clip(score + noise, 0, 1)
placed    = (prob > 0.52).astype(int)

# ── Build DataFrame ───────────────────────────────────────────────────────────
df = pd.DataFrame({
    'cgpa'           : cgpa,
    'marks_10th'     : marks_10th,
    'marks_12th'     : marks_12th,
    'internships'    : internships,
    'projects'       : projects,
    'certifications' : certifications,
    'backlogs'       : backlogs,
    'skills_score'   : skills_score,
    'communication'  : communication,
    'branch'         : branch,
    'gender'         : gender,
    'placed'         : placed
})

df.to_csv('data/students.csv', index=False)

print("Dataset created: data/students.csv")
print(f"Shape          : {df.shape}")
print(f"Placed         : {placed.sum()} / {n}  ({placed.mean()*100:.1f}%)")
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)