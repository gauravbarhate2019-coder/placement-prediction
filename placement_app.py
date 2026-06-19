"""
STREAMLIT WEB APP — Student Placement Predictor
With 3D Pie Chart + Summary Card
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

    result_text = "✅ PLACED" if prediction == 1 else "❌ NOT PLACED"
    card_color  = "#1B5E20" if prediction == 1 else "#B71C1C"
    badge_color = "#4CAF50" if prediction == 1 else "#F44336"

    # ── Summary Card ──────────────────────────────────────────────────────────
    st.markdown("### 🪪 Student Summary Card")
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {card_color}, #1a1a2e);
        border: 2px solid {badge_color};
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    ">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
            <div>
                <h2 style="margin:0; color:white; font-size:24px;">🎓 {name if name else 'Student'}</h2>
                <p style="margin:4px 0 0 0; color:#ccc; font-size:14px;">🏫 {college if college else 'College'} &nbsp;|&nbsp; {branch} &nbsp;|&nbsp; {gender}</p>
            </div>
            <div style="
                background:{badge_color};
                color:white;
                padding:8px 18px;
                border-radius:20px;
                font-weight:bold;
                font-size:15px;
            ">{result_text}</div>
        </div>

        <hr style="border-color:rgba(255,255,255,0.15); margin:12px 0;">

        <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-top:14px;">
            <div style="background:rgba(255,255,255,0.08); border-radius:10px; padding:12px; text-align:center;">
                <div style="font-size:22px; font-weight:bold; color:white;">{cgpa}</div>
                <div style="color:#aaa; font-size:12px;">CGPA</div>
            </div>
            <div style="background:rgba(255,255,255,0.08); border-radius:10px; padding:12px; text-align:center;">
                <div style="font-size:22px; font-weight:bold; color:white;">{int(prob_placed)}%</div>
                <div style="color:#aaa; font-size:12px;">Placement Score</div>
            </div>
            <div style="background:rgba(255,255,255,0.08); border-radius:10px; padding:12px; text-align:center;">
                <div style="font-size:22px; font-weight:bold; color:white;">{level}</div>
                <div style="color:#aaa; font-size:12px;">Readiness</div>
            </div>
            <div style="background:rgba(255,255,255,0.08); border-radius:10px; padding:12px; text-align:center;">
                <div style="font-size:22px; font-weight:bold; color:white;">{internships}</div>
                <div style="color:#aaa; font-size:12px;">Internships</div>
            </div>
            <div style="background:rgba(255,255,255,0.08); border-radius:10px; padding:12px; text-align:center;">
                <div style="font-size:22px; font-weight:bold; color:white;">{projects}</div>
                <div style="color:#aaa; font-size:12px;">Projects</div>
            </div>
            <div style="background:rgba(255,255,255,0.08); border-radius:10px; padding:12px; text-align:center;">
                <div style="font-size:22px; font-weight:bold; color:white;">{skills_score}/10</div>
                <div style="color:#aaa; font-size:12px;">Skills Score</div>
            </div>
            <div style="background:rgba(255,255,255,0.08); border-radius:10px; padding:12px; text-align:center;">
                <div style="font-size:22px; font-weight:bold; color:white;">{marks_10th}%</div>
                <div style="color:#aaa; font-size:12px;">10th Marks</div>
            </div>
            <div style="background:rgba(255,255,255,0.08); border-radius:10px; padding:12px; text-align:center;">
                <div style="font-size:22px; font-weight:bold; color:white;">{marks_12th}%</div>
                <div style="color:#aaa; font-size:12px;">12th Marks</div>
            </div>
            <div style="background:rgba(255,255,255,0.08); border-radius:10px; padding:12px; text-align:center;">
                <div style="font-size:22px; font-weight:bold; color:white;">{backlogs}</div>
                <div style="color:#aaa; font-size:12px;">Backlogs</div>
            </div>
        </div>

        <hr style="border-color:rgba(255,255,255,0.15); margin:16px 0 10px 0;">
        <p style="color:#aaa; font-size:12px; margin:0; text-align:right;">
            🤖 Predicted by Logistic Regression &nbsp;|&nbsp; Accuracy: 83%
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Standard result ───────────────────────────────────────────────────────
    st.markdown("### 📊 Prediction Result")
    if prediction == 1:
        st.success(f"✅ **PLACED** — {prob_placed:.1f}% confidence")
    else:
        st.error(f"❌ **NOT PLACED** — {prob_placed:.1f}% placement chance")

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
        marker=dict(colors=['#4CAF50', '#F44336'], line=dict(color='white', width=2)),
        textinfo='label+percent',
        textfont=dict(size=15, color='white'),
        rotation=135,
        direction='clockwise',
    )])
    fig.update_layout(
        title=dict(text=f"Placement Probability — {name if name else 'Student'}", x=0.5, font=dict(size=15)),
        showlegend=True,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        height=400,
        margin=dict(t=60, b=80, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
    )
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