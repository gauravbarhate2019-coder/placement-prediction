"""
STREAMLIT WEB APP — Student Placement Predictor
Run with: streamlit run app.py
"""

import pickle
import numpy as np
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="centered"
)

# ── Load model & preprocessor ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    import os
    if not os.path.exists('models/preprocessed.pkl'):
        # Auto-generate model if not found
        import subprocess
        subprocess.run(['python', 'step1_generate_data.py'])
        subprocess.run(['python', 'step3_preprocess.py'])
        subprocess.run(['python', 'step4_train_models.py'])
    
    with open('models/preprocessed.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('models/best_model.pkl', 'rb') as f:
        best = pickle.load(f)
    return data, best

return data, best

# Load model and preprocessing objects
data, model = load_model()

scaler = data['scaler']
le_branch = data['le_branch']
le_gender = data['le_gender']

# Header
st.title("🎓 Student Placement Predictor")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎓 Student Placement Predictor")
st.markdown(f"**Model:** Logistic Regression &nbsp;|&nbsp; **Accuracy:** 83%")
st.markdown("---")

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("📋 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    cgpa         = st.slider("CGPA",            5.0, 10.0, 7.5, 0.1)
    marks_10th   = st.slider("10th Marks (%)",  50.0, 100.0, 75.0, 0.5)
    marks_12th   = st.slider("12th Marks (%)",  50.0, 100.0, 75.0, 0.5)
    internships  = st.selectbox("Internships",  [0, 1, 2, 3])
    projects     = st.selectbox("Projects",     [0, 1, 2, 3, 4, 5])

with col2:
    certifications = st.selectbox("Certifications", [0, 1, 2, 3, 4])
    backlogs       = st.selectbox("Backlogs",       [0, 1, 2, 3, 4])
    skills_score   = st.slider("Technical Skills (1–10)", 1, 10, 6)
    communication  = st.slider("Communication  (1–10)", 1, 10, 6)
    branch         = st.selectbox("Branch", ['CSE', 'ECE', 'IT', 'Mech', 'Civil'])
    gender         = st.selectbox("Gender", ['Male', 'Female'])

st.markdown("---")

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔮 Predict Placement", use_container_width=True):

    branch_enc = le_branch.transform([branch])[0]
    gender_enc = le_gender.transform([gender])[0]

    features = np.array([[cgpa, marks_10th, marks_12th, internships, projects,
                          certifications, backlogs, skills_score, communication,
                          branch_enc, gender_enc]])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    prob_placed = probability[1] * 100

    # ── Result ────────────────────────────────────────────────────────────────
    st.markdown("### 📊 Prediction Result")

    if prediction == 1:
        st.success(f"✅ **PLACED** — {prob_placed:.1f}% confidence")
    else:
        st.error(f"❌ **NOT PLACED** — {prob_placed:.1f}% placement chance")

    # Probability bar
    st.markdown("**Placement Probability**")
    st.progress(int(prob_placed))
    col_a, col_b = st.columns(2)
    col_a.metric("Placed",     f"{prob_placed:.1f}%")
    col_b.metric("Not Placed", f"{100 - prob_placed:.1f}%")

    # ── Advice ────────────────────────────────────────────────────────────────
    advice = []
    if cgpa < 7.0:          advice.append("📈 Improve CGPA above 7.0")
    if internships == 0:    advice.append("💼 Do at least 1 internship")
    if skills_score < 7:    advice.append("💻 Increase technical skills")
    if backlogs > 1:        advice.append("📚 Clear your backlogs")
    if communication < 6:   advice.append("🗣️ Work on communication skills")
    if projects < 2:        advice.append("🛠️ Build more projects (aim for 3+)")

    st.markdown("### 💡 Improvement Advice")
    if advice:
        for tip in advice:
            st.warning(tip)
    else:
        st.success("🌟 You look well-prepared! Keep it up.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with Python · scikit-learn · Streamlit · Internship Project")