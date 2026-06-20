"""
STREAMLIT WEB APP — Student Placement Predictor
Kick to Tech | With Logo + Summary Card + 3D Pie + Feature Impact + Download
"""

import os
import io
import base64
import pickle
import subprocess
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Student Placement Predictor | Kick to Tech",
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

def get_logo_base64():
    try:
        with open('logo.png', 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

data, model = load_model()
scaler    = data['scaler']
le_branch = data['le_branch']
le_gender = data['le_gender']
logo_b64  = get_logo_base64()

# ── Header with Kick to Tech logo ─────────────────────────────────────────────
if logo_b64:
    st.markdown(f"""
    <div style="
        display: flex;
        align-items: center;
        gap: 18px;
        background: linear-gradient(135deg, #0d7377, #14a085);
        border-radius: 14px;
        padding: 18px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    ">
        <img src="data:image/png;base64,{logo_b64}"
             style="width:90px; height:90px; border-radius:10px; object-fit:cover;"/>
        <div>
            <h1 style="margin:0; color:white; font-size:26px; font-weight:bold;">
                Kick to Tech
            </h1>
            <p style="margin:4px 0 0 0; color:#d0f0ec; font-size:14px;">
                🎓 Student Placement Prediction System
            </p>
            <p style="margin:2px 0 0 0; color:#a8e6df; font-size:12px;">
                Model: Logistic Regression &nbsp;|&nbsp; Accuracy: 83%
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.title("🎓 Student Placement Predictor | Kick to Tech")
    st.markdown("**Model:** Logistic Regression &nbsp;|&nbsp; **Accuracy:** 83%")

st.markdown("---")

# ── Input form ────────────────────────────────────────────────────────────────
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

    if prob_placed >= 80:   level = "Excellent"
    elif prob_placed >= 60: level = "Good"
    elif prob_placed >= 40: level = "Average"
    else:                   level = "Needs Improvement"

    result_text = "PLACED" if prediction == 1 else "NOT PLACED"
    card_color  = "#1B5E20" if prediction == 1 else "#7f0000"
    badge_color = "#4CAF50" if prediction == 1 else "#F44336"
    emoji       = "✅" if prediction == 1 else "❌"

    # ── Summary Card ──────────────────────────────────────────────────────────
    st.markdown("### 🪪 Student Summary Card")
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="width:40px;height:40px;border-radius:6px;vertical-align:middle;margin-right:8px;"/>' if logo_b64 else ""
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{card_color},#1a1a2e);
                border:2px solid {badge_color}; border-radius:16px;
                padding:24px 28px 10px 28px; box-shadow:0 4px 20px rgba(0,0,0,0.4);">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <div>
                <h2 style="margin:0; color:white; font-size:22px;">
                    🎓 {name if name else 'Student'}
                </h2>
                <p style="margin:4px 0 0 0; color:#ccc; font-size:13px;">
                    🏫 {college if college else 'College'} &nbsp;|&nbsp; {branch} &nbsp;|&nbsp; {gender}
                </p>
                <p style="margin:4px 0 0 0; color:#aaa; font-size:12px;">
                    {logo_html} Kick to Tech Internship Project
                </p>
            </div>
            <div style="background:{badge_color}; color:white; padding:8px 18px;
                        border-radius:20px; font-weight:bold; font-size:15px;">
                {emoji} {result_text}
            </div>
        </div>
        <hr style="border-color:rgba(255,255,255,0.2); margin:12px 0 6px 0;">
        <p style="color:#aaa; font-size:12px; margin:0; text-align:right;">
            🤖 Logistic Regression &nbsp;|&nbsp; Accuracy: 83%
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("📊 CGPA",           f"{cgpa}")
    c2.metric("🎯 Placement Score", f"{int(prob_placed)}/100")
    c3.metric("📈 Readiness",       level)

    c4, c5, c6 = st.columns(3)
    c4.metric("💼 Internships",    f"{internships}")
    c5.metric("🛠️ Projects",       f"{projects}")
    c6.metric("💻 Skills Score",   f"{skills_score}/10")

    c7, c8, c9 = st.columns(3)
    c7.metric("📝 10th Marks",     f"{marks_10th}%")
    c8.metric("📝 12th Marks",     f"{marks_12th}%")
    c9.metric("⚠️ Backlogs",       f"{backlogs}")

    st.markdown("---")

    # ── Prediction result ─────────────────────────────────────────────────────
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
    fig_pie = go.Figure(data=[go.Pie(
        labels=['Placed', 'Not Placed'],
        values=[prob_placed, prob_not_placed],
        pull=[0.1, 0],
        marker=dict(colors=['#4CAF50', '#F44336'], line=dict(color='white', width=2)),
        textinfo='label+percent',
        textfont=dict(size=15, color='white'),
        rotation=135, direction='clockwise',
    )])
    fig_pie.update_layout(
        title=dict(text=f"Placement Probability — {name if name else 'Student'}",
                   x=0.5, font=dict(size=15)),
        showlegend=True,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        height=400, margin=dict(t=60, b=80, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # ── Feature Impact Chart ───────────────────────────────────────────────────
    st.markdown("### 📉 Feature Impact Chart")
    feature_labels = ['CGPA','10th Marks','12th Marks','Internships',
                      'Projects','Certifications','Skills Score','Communication','No Backlogs']
    raw_scores = [
        (cgpa - 5) / 5 * 10,
        (marks_10th - 50) / 50 * 10,
        (marks_12th - 50) / 50 * 10,
        internships / 3 * 10,
        projects / 5 * 10,
        certifications / 4 * 10,
        skills_score,
        communication,
        max(0, (4 - backlogs) / 4 * 10),
    ]
    scores = [round(min(10, max(0, s)), 1) for s in raw_scores]
    bar_colors = ['#4CAF50' if s >= 7 else '#FF9800' if s >= 5 else '#F44336' for s in scores]

    fig_impact = go.Figure()
    fig_impact.add_trace(go.Bar(
        name='Ideal Score', x=feature_labels, y=[10]*len(scores),
        marker_color='rgba(255,255,255,0.1)',
        marker_line=dict(color='rgba(255,255,255,0.3)', width=1),
    ))
    fig_impact.add_trace(go.Bar(
        name='Your Score', x=feature_labels, y=scores,
        marker_color=bar_colors,
        marker_line=dict(color='white', width=1),
        text=[f"{s}" for s in scores],
        textposition='outside',
        textfont=dict(color='white', size=11),
    ))
    fig_impact.update_layout(
        barmode='overlay',
        title=dict(text=f"Feature Impact — {name if name else 'Student'}",
                   x=0.5, font=dict(size=15)),
        xaxis=dict(tickangle=-25, tickfont=dict(size=11)),
        yaxis=dict(range=[0, 12], title="Score (0-10)"),
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
        height=420, margin=dict(t=60, b=100, l=40, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_gridcolor='rgba(255,255,255,0.1)',
    )
    st.plotly_chart(fig_impact, use_container_width=True)

    col_g, col_o, col_r = st.columns(3)
    col_g.success("🟢 Strong (7-10)")
    col_o.warning("🟠 Average (5-6)")
    col_r.error("🔴 Weak (0-4)")

    # ── Advice ────────────────────────────────────────────────────────────────
    advice = []
    if cgpa < 7.0:         advice.append("Improve CGPA above 7.0")
    if internships == 0:   advice.append("Do at least 1 internship")
    if skills_score < 7:   advice.append("Increase technical skills score")
    if backlogs > 1:       advice.append("Clear your backlogs")
    if communication < 6:  advice.append("Work on communication skills")
    if projects < 2:       advice.append("Build more projects (aim for 3+)")

    st.markdown("### 💡 Improvement Advice")
    if advice:
        for tip in advice:
            st.warning(f"• {tip}")
    else:
        st.success("🌟 You look well-prepared! Keep it up.")

    # ── Download Report ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📥 Download Report")
    now = datetime.now().strftime("%d-%m-%Y %H:%M")
    advice_text = "\n".join([f"  • {a}" for a in advice]) if advice else "  • You are well-prepared!"

    report = f"""
╔══════════════════════════════════════════════════════════╗
║          STUDENT PLACEMENT PREDICTION REPORT            ║
║                   Kick to Tech                          ║
╚══════════════════════════════════════════════════════════╝

Generated On  : {now}
Model Used    : Logistic Regression
Model Accuracy: 83%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STUDENT INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name          : {name if name else 'N/A'}
College       : {college if college else 'N/A'}
Branch        : {branch}
Gender        : {gender}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACADEMIC PROFILE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CGPA          : {cgpa}
10th Marks    : {marks_10th}%
12th Marks    : {marks_12th}%
Backlogs      : {backlogs}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SKILLS & EXPERIENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Internships   : {internships}
Projects      : {projects}
Certifications: {certifications}
Skills Score  : {skills_score}/10
Communication : {communication}/10

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PREDICTION RESULT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Result        : {emoji} {result_text}
Placed        : {prob_placed:.1f}%
Not Placed    : {prob_not_placed:.1f}%
Placement Score: {int(prob_placed)}/100
Readiness     : {level}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FEATURE SCORES (out of 10)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{chr(10).join([f"  {feature_labels[i]:<18}: {scores[i]:>4}/10  {'✅' if scores[i]>=7 else '⚠️' if scores[i]>=5 else '❌'}" for i in range(len(scores))])}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPROVEMENT ADVICE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{advice_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Kick to Tech | Student Placement Prediction System
Built with Python · scikit-learn · Streamlit · Plotly
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    filename = f"placement_report_{(name if name else 'student').replace(' ','_')}.txt"
    st.download_button(
        label="📥 Download Placement Report",
        data=report,
        file_name=filename,
        mime="text/plain",
        use_container_width=True
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
if logo_b64:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:10px; justify-content:center; opacity:0.7;">
        <img src="data:image/png;base64,{logo_b64}" style="width:28px; height:28px; border-radius:4px;"/>
        <span style="color:#aaa; font-size:12px;">Kick to Tech · Student Placement Prediction · Built with Python & Streamlit</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.caption("Kick to Tech · Built with Python · scikit-learn · Streamlit · Plotly")