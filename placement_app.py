"""
STREAMLIT WEB APP — Student Placement Predictor
With 3D Pie Chart using Plotly
"""

import os
import pickle
import subprocess
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="centered"
)

@st.cache_resource
def load_model():
    if not os.path.exists('models/preprocessed.pkl'):
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('outputs', exist_ok=True)
        subprocess.run(['python', 'step1_generate_data.py'])
        subprocess.run(['python', 'step3_preprocess.py'])
        subprocess.run(['python', 'step4_train_models.py'])

    with open('models/preprocessed.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('models/best_model.pkl', 'rb') as f:
        best = pickle.load(f)
    return data, best['model']

data, model = load_model()
scaler    = data['scaler']
le_branch = data['le_branch']
le_gender = data['le_gender']

st.title("🎓 Student Placement Predictor")
st.markdown("**Model:** Logistic Regression &nbsp;|&nbsp; **Accuracy:** 83%")
st.markdown("---")

st.subheader("📋 Enter Student Details")
name    = st.text_input("Student Name")
college = st.text_input("College Name")

col1, col2 = st.columns(2)
with col1:
    cgpa         = st.slider("CGPA",           5.0, 10.0, 7.5, 0.1)
    marks_10th   = st.slider("10th Marks (%)", 50.0, 100.0, 75.0, 0.5)
    marks_12th   = st.slider("12th Marks (%)", 50.0, 100.0, 75.0, 0.5)
    internships  = st.selectbox("Internships", [0, 1, 2, 3])
    projects     = st.selectbox("Projects",    [0, 1, 2, 3, 4, 5])

with col2:
    certifications = st.selectbox("Certifications", [0, 1, 2, 3, 4])
    backlogs       = st.selectbox("Backlogs",       [0, 1, 2, 3, 4])
    skills_score   = st.slider("Technical Skills (1-10)", 1, 10, 6)
    communication  = st.slider("Communication (1-10)",    1, 10, 6)
    branch         = st.selectbox("Branch", ['CSE', 'ECE', 'IT', 'Mech', 'Civil'])
    gender         = st.selectbox("Gender", ['Male', 'Female'])

st.markdown("---")

if st.button("🔮 Predict Placement", use_container_width=True):

    branch_enc = le_branch.transform([branch])[0]
    gender_enc = le_gender.transform([gender])[0]

    features = np.array([[cgpa, marks_10th, marks_12th, internships, projects,
                          certifications, backlogs, skills_score, communication,
                          branch_enc, gender_enc]])
    features_scaled = scaler.transform(features)

    prediction      = model.predict(features_scaled)[0]
    probability     = model.predict_proba(features_scaled)[0]
    prob_placed     = probability[1] * 100
    prob_not_placed = probability[0] * 100

    if prob_placed >= 80:   level = "🌟 Excellent"
    elif prob_placed >= 60: level = "✅ Good"
    elif prob_placed >= 40: level = "⚠️ Average"
    else:                   level = "❌ Needs Improvement"

    st.markdown("### 📊 Prediction Result")
    if name:    st.write(f"👨 Student : **{name}**")
    if college: st.write(f"🏫 College : **{college}**")

    st.info(f"Placement Readiness : {level}")
    st.metric("🎯 Placement Score", f"{int(prob_placed)}/100")

    if prediction == 1:
        st.success(f"✅ **PLACED** — {prob_placed:.1f}% confidence")
    else:
        st.error(f"❌ **NOT PLACED** — {prob_placed:.1f}% placement chance")

    st.markdown("**Placement Probability**")
    st.progress(int(prob_placed))
    col_a, col_b = st.columns(2)
    col_a.metric("Placed",     f"{prob_placed:.1f}%")
    col_b.metric("Not Placed", f"{prob_not_placed:.1f}%")

    # ── 3D Pie Chart ──────────────────────────────────────────────────────────
    st.markdown("### 🥧 3D Placement Probability Chart")

    fig = go.Figure(data=[go.Pie(
        labels=['Placed', 'Not Placed'],
        values=[prob_placed, prob_not_placed],
        pull=[0.1, 0],
        marker=dict(
            colors=['#4CAF50', '#F44336'],
            line=dict(color='white', width=2)
        ),
        textinfo='label+percent',
        textfont=dict(size=15, color='white'),
        rotation=135,
        direction='clockwise',
    )])

    fig.update_layout(
        title=dict(
            text=f"Placement Probability — {name if name else 'Student'}",
            x=0.5,
            font=dict(size=15)
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        height=420,
        margin=dict(t=60, b=80, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        annotations=[dict(
            text=f"<b>{int(prob_placed)}%</b><br>Placed",
            x=0.5, y=0.5,
            font=dict(size=18, color='white'),
            showarrow=False
        )]
    )

    # 3D effect — add outer ring shadow
    fig.add_trace(go.Pie(
        labels=['Placed', 'Not Placed'],
        values=[prob_placed, prob_not_placed],
        pull=[0.1, 0],
        marker=dict(colors=['#2E7D32', '#B71C1C'], line=dict(color='white', width=1)),
        textinfo='none',
        rotation=135,
        direction='clockwise',
        opacity=0.35,
        domain=dict(x=[0.02, 0.98], y=[0.0, 0.07]),
        showlegend=False
    ))

    st.plotly_chart(fig, use_container_width=True)

    # ── Advice ────────────────────────────────────────────────────────────────
    advice = []
    if cgpa < 7.0:         advice.append("📈 Improve CGPA above 7.0")
    if internships == 0:   advice.append("💼 Do at least 1 internship")
    if skills_score < 7:   advice.append("💻 Increase technical skills")
    if backlogs > 1:       advice.append("📚 Clear your backlogs")
    if communication < 6:  advice.append("🗣️ Work on communication skills")
    if projects < 2:       advice.append("🛠️ Build more projects (aim for 3+)")

    st.markdown("### 💡 Improvement Advice")
    if advice:
        for tip in advice:
            st.warning(tip)
    else:
        st.success("🌟 You look well-prepared! Keep it up.")

st.markdown("---")
st.caption("Built with Python · scikit-learn · Streamlit · Plotly · Internship Project")