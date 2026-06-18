"""
STEP 6 — MAKE PREDICTIONS
Load the saved model and predict placement for a new student.
"""

import numpy as np
import pickle

# ── Load model and preprocessor ───────────────────────────────────────────────
with open('models/preprocessed.pkl', 'rb') as f:
    data = pickle.load(f)
with open('models/best_model.pkl', 'rb') as f:
    best = pickle.load(f)

scaler    = data['scaler']
le_branch = data['le_branch']
le_gender = data['le_gender']
model     = best['model']
FEATURES  = data['feature_names']


def predict_placement(cgpa, marks_10th, marks_12th, internships,
                      projects, certifications, backlogs,
                      skills_score, communication, branch, gender):
    """
    Predict whether a student will be placed.

    Parameters
    ----------
    cgpa            : float  (5.0 – 10.0)
    marks_10th      : float  (50 – 100)
    marks_12th      : float  (50 – 100)
    internships     : int    (0 – 3)
    projects        : int    (0 – 5)
    certifications  : int    (0 – 4)
    backlogs        : int    (0 – 4)
    skills_score    : int    (1 – 10)
    communication   : int    (1 – 10)
    branch          : str    ('CSE','ECE','Mech','Civil','IT')
    gender          : str    ('Male','Female')

    Returns
    -------
    dict with prediction, probability, and advice
    """
    branch_enc = le_branch.transform([branch])[0]
    gender_enc = le_gender.transform([gender])[0]

    features = np.array([[cgpa, marks_10th, marks_12th, internships, projects,
                          certifications, backlogs, skills_score, communication,
                          branch_enc, gender_enc]])

    features_scaled = scaler.transform(features)

    prediction  = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]

    prob_placed     = probability[1] * 100
    prob_not_placed = probability[0] * 100

    result = "✅ PLACED" if prediction == 1 else "❌ NOT PLACED"

    # Simple advice engine
    advice = []
    if cgpa < 7.0:         advice.append("Improve CGPA above 7.0")
    if internships == 0:   advice.append("Do at least 1 internship")
    if skills_score < 7:   advice.append("Increase technical skills score")
    if backlogs > 1:       advice.append("Clear your backlogs")
    if communication < 6:  advice.append("Work on communication skills")
    if projects < 2:       advice.append("Build more projects (aim for 3+)")

    return {
        'prediction'    : result,
        'placed_prob'   : f'{prob_placed:.1f}%',
        'not_placed_prob': f'{prob_not_placed:.1f}%',
        'advice'        : advice if advice else ['You look well-prepared! Keep it up.']
    }


# ── Test with 3 student profiles ──────────────────────────────────────────────
print("=" * 55)
print("PLACEMENT PREDICTION SYSTEM")
print("=" * 55)

students = [
    {
        "name"          : "Student A — Strong profile",
        "cgpa"          : 8.9, "marks_10th": 88, "marks_12th": 85,
        "internships"   : 3, "projects": 4, "certifications": 3,
        "backlogs"      : 0, "skills_score": 9, "communication": 8,
        "branch"        : "CSE", "gender": "Male"
    },
    {
        "name"          : "Student B — Average profile",
        "cgpa"          : 6.8, "marks_10th": 65, "marks_12th": 67,
        "internships"   : 1, "projects": 2, "certifications": 1,
        "backlogs"      : 2, "skills_score": 6, "communication": 6,
        "branch"        : "ECE", "gender": "Female"
    },
    {
        "name"          : "Student C — Weak profile",
        "cgpa"          : 5.5, "marks_10th": 52, "marks_12th": 55,
        "internships"   : 0, "projects": 1, "certifications": 0,
        "backlogs"      : 4, "skills_score": 4, "communication": 4,
        "branch"        : "Mech", "gender": "Male"
    }
]

for s in students:
    name = s.pop("name")
    result = predict_placement(**s)

    print(f"\n{name}")
    print(f"  Prediction : {result['prediction']}")
    print(f"  Placed     : {result['placed_prob']}")
    print(f"  Not placed : {result['not_placed_prob']}")
    print(f"  Advice     :")
    for tip in result['advice']:
        print(f"    • {tip}")

print("\n" + "="*55)
print("Model is ready! Edit the student dict to predict any student.")