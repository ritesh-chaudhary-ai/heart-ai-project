import streamlit as st
import pandas as pd
import joblib
import base64
import io
import re
import math
from datetime import datetime

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Stroke Predictor",
    layout="wide",
    page_icon="🫀",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: #020B18;
    color: #D1D9E6;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.2rem 2rem; max-width: 1440px; }

/* ── Sidebar always visible fix ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #040F1E 0%, #081525 100%);
    border-right: 1px solid #0F2135;
    display: block !important;
    visibility: visible !important;
    min-width: 21rem !important;
    transform: none !important;
    transition: none !important;
}
[data-testid="stSidebar"][aria-expanded="false"] {
    display: block !important;
    visibility: visible !important;
    width: 21rem !important;
    min-width: 21rem !important;
    transform: none !important;
}
[data-testid="stSidebarCollapsedControl"],
button[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 { color: #00E5C3; }

.cardio-card {
    background: linear-gradient(145deg, #081525 0%, #0C1D30 100%);
    border: 1px solid #0F2840;
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 2px 20px rgba(0,0,0,0.5);
    transition: border-color 0.3s ease;
}
.cardio-card:hover { border-color: #003D5A; }

.badge-high {
    background: linear-gradient(90deg, #7B0020, #B01535);
    color: #FFD6DE;
    padding: 0.35rem 1.1rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 0.82rem;
    letter-spacing: 0.06em;
    display: inline-block;
    font-family: 'Syne', sans-serif;
}
.badge-low {
    background: linear-gradient(90deg, #064E3B, #047857);
    color: #CCFCE8;
    padding: 0.35rem 1.1rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 0.82rem;
    letter-spacing: 0.06em;
    display: inline-block;
    font-family: 'Syne', sans-serif;
}
.badge-moderate {
    background: linear-gradient(90deg, #78350F, #B45309);
    color: #FEF3C7;
    padding: 0.35rem 1.1rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 0.82rem;
    letter-spacing: 0.06em;
    display: inline-block;
    font-family: 'Syne', sans-serif;
}

.metric-tile {
    background: linear-gradient(135deg, #040F1E, #081525);
    border: 1px solid #0F2840;
    border-radius: 10px;
    padding: 0.9rem;
    text-align: center;
}
.metric-value {
    font-size: 1.7rem;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    color: #4A6080;
    text-transform: uppercase;
    margin-top: 0.2rem;
    font-family: 'Outfit', sans-serif;
}

.stSlider [data-baseweb="slider"] { color: #00E5C3; }
.stSelectbox [data-baseweb="select"] > div,
.stNumberInput input,
.stTextInput input {
    background: #040F1E !important;
    border: 1px solid #0F2840 !important;
    border-radius: 7px !important;
    color: #D1D9E6 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

.stButton > button {
    background: linear-gradient(90deg, #007A6A, #00E5C3);
    color: #020B18;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
    letter-spacing: 0.06em;
    border: none;
    border-radius: 9px;
    padding: 0.6rem 1.5rem;
    transition: all 0.2s ease;
    width: 100%;
    font-size: 0.9rem;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #00E5C3, #67ECD6);
    transform: translateY(-1px);
    box-shadow: 0 0 22px rgba(0,229,195,0.25);
}

.chat-user {
    background: linear-gradient(135deg, #0F2840, #0C1D30);
    border: 1px solid #1A3A5C;
    border-radius: 14px 14px 4px 14px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0 0.4rem 3rem;
    font-size: 0.875rem;
    line-height: 1.5;
}
.chat-ai {
    background: linear-gradient(135deg, #04180F, #081A12);
    border: 1px solid #0D4A30;
    border-radius: 14px 14px 14px 4px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 3rem 0.4rem 0;
    font-size: 0.875rem;
    line-height: 1.6;
}
.chat-ai-label {
    color: #00E5C3;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    margin-bottom: 0.3rem;
    font-family: 'Syne', sans-serif;
}
.chat-user-label {
    color: #4A6080;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    margin-bottom: 0.3rem;
    text-align: right;
    font-family: 'Syne', sans-serif;
}
.section-header {
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #00E5C3;
    font-weight: 700;
    border-bottom: 1px solid #0F2840;
    padding-bottom: 0.45rem;
    margin-bottom: 0.9rem;
    font-family: 'Syne', sans-serif;
}
.insight-pill {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 50px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 0.15rem;
    font-family: 'IBM Plex Mono', monospace;
}
.pill-danger { background: rgba(176,21,53,0.25); border: 1px solid #B01535; color: #FFD6DE; }
.pill-warn   { background: rgba(180,83,9,0.25);  border: 1px solid #B45309; color: #FEF3C7; }
.pill-ok     { background: rgba(4,120,87,0.2);   border: 1px solid #047857; color: #CCFCE8; }

.ai-avatar {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    background: linear-gradient(135deg, #007A6A, #00E5C3);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    border: 2px solid rgba(0,229,195,0.5);
    box-shadow: 0 0 14px rgba(0,229,195,0.2);
}

hr { border-color: #0F2840 !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #040F1E; }
::-webkit-scrollbar-thumb { background: #0F2840; border-radius: 2px; }

.stTabs [data-baseweb="tab-list"] { background: #040F1E; border-radius: 10px; gap: 4px; padding: 4px; }
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 7px;
    color: #4A6080;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 0.05em;
    padding: 0.4rem 1rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #007A6A22, #00E5C322) !important;
    color: #00E5C3 !important;
    border-bottom: 2px solid #00E5C3 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
defaults = {
    "result": None,
    "probability": None,
    "patient_data": {},
    "patient_name": "Anonymous",
    "chat_history": [],
    "prediction_done": False,
    "clinical_summary": "",
    "risk_flags": [],
    "recommendations": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        data = joblib.load("heart_app.pkl")
        return data["model"], data["scaler"], data["columns"]
    except Exception as e:
        st.error(f"❌ Model load error: {e}")
        st.stop()

model, scaler, columns = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# LABEL MAPS
# ─────────────────────────────────────────────────────────────────────────────
CP_LABELS   = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-Anginal Pain", 3: "Asymptomatic"}
ECG_LABELS  = {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}
SLOPE_LBLS  = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
THAL_LABELS = {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect", 3: "Unknown"}
SEX_LABELS  = {0: "Female", 1: "Male"}

# ─────────────────────────────────────────────────────────────────────────────
# AI AVATAR SVG
# ─────────────────────────────────────────────────────────────────────────────
AI_AVATAR_SVG = """
<svg width="26" height="26" viewBox="0 0 26 26" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect x="5" y="9" width="16" height="12" rx="3" fill="#020B18" opacity="0.9"/>
  <rect x="8.5" y="12.5" width="3" height="3" rx="0.8" fill="#00E5C3"/>
  <rect x="14.5" y="12.5" width="3" height="3" rx="0.8" fill="#00E5C3"/>
  <rect x="10" y="16.5" width="6" height="1.8" rx="0.9" fill="#00E5C3"/>
  <rect x="11" y="5" width="4" height="4.5" rx="1.2" fill="#020B18" opacity="0.9"/>
  <circle cx="13" cy="4.5" r="2" fill="#00E5C3"/>
  <rect x="1.5" y="11.5" width="2.5" height="6" rx="1.2" fill="#020B18" opacity="0.8"/>
  <rect x="22" y="11.5" width="2.5" height="6" rx="1.2" fill="#020B18" opacity="0.8"/>
  <line x1="4" y1="13" x2="5" y2="14" stroke="#00E5C3" stroke-width="1.2" stroke-linecap="round"/>
  <line x1="22" y1="13" x2="21" y2="14" stroke="#00E5C3" stroke-width="1.2" stroke-linecap="round"/>
</svg>
"""

def ai_bubble(content_html):
    return f"""
    <div class='chat-ai'>
        <div style='display:flex;align-items:center;gap:0.65rem;margin-bottom:0.55rem;'>
            <div class='ai-avatar'>{AI_AVATAR_SVG}</div>
            <span class='chat-ai-label' style='margin-bottom:0;'>HEART STROKE PREDICTOR AI</span>
        </div>
        {content_html}
    </div>"""

def user_bubble(content_html):
    return f"""
    <div class='chat-user'>
        <div class='chat-user-label'>YOU</div>
        {content_html}
    </div>"""

# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED RULE-BASED CLINICAL AI ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class ClinicalEngine:
    """Rule-based clinical decision support engine."""

    def __init__(self, pd_: dict, name: str, result: str, prob: int):
        self.p    = pd_
        self.name = name
        self.res  = result
        self.prob = prob

    def _bp_level(self):
        bp = self.p["trestbps"]
        if bp < 120: return "normal", "ok"
        if bp < 130: return "elevated", "warn"
        if bp < 140: return "Stage-1 hypertension", "warn"
        return "Stage-2 hypertension", "danger"

    def _chol_level(self):
        c = self.p["chol"]
        if c < 200: return "desirable", "ok"
        if c < 240: return "borderline-high", "warn"
        return "high", "danger"

    def _hr_reserve(self):
        pred_max = 220 - self.p["age"]
        reserve  = (self.p["thalach"] / pred_max) * 100
        return reserve

    def _framingham_risk_estimate(self):
        p = self.p
        score = 0
        age = p["age"]
        if p["sex"] == 1:
            score += max(0, (age - 30) // 5 * 2)
            if p["chol"] >= 240: score += 3
            if p["trestbps"] >= 140: score += 2
            if p["fbs"] == 1: score += 2
        else:
            score += max(0, (age - 30) // 5 * 2 - 1)
            if p["chol"] >= 240: score += 2
            if p["trestbps"] >= 140: score += 4
        if p["exang"] == 1: score += 3
        if p["oldpeak"] > 2: score += 2
        if p["ca"] >= 2: score += 4
        return min(score * 2, 40)

    def generate_summary(self) -> str:
        p    = self.p
        bp_l, _ = self._bp_level()
        ch_l, _ = self._chol_level()
        hr_res   = self._hr_reserve()
        fram     = self._framingham_risk_estimate()

        sex_str = SEX_LABELS[p["sex"]]
        cp_str  = CP_LABELS[p["cp"]]
        ecg_str = ECG_LABELS[p["restecg"]]
        slp_str = SLOPE_LBLS[p["slope"]]
        thl_str = THAL_LABELS[p["thal"]]

        risk_word = "HIGH" if self.prob >= 65 else "MODERATE" if self.prob >= 40 else "LOW"

        lines = [
            f"**Patient:** {self.name} | {sex_str}, Age {p['age']}",
            f"**Model Output:** {self.res} ({self.prob}% probability) — {risk_word} RISK TIER",
            "",
            "---",
            "**📋 Clinical Findings Summary**",
            "",
        ]

        lines.append(f"• **Blood Pressure:** {p['trestbps']} mmHg — classified as *{bp_l}*. "
                     + ("Urgent antihypertensive review recommended." if bp_l == "Stage-2 hypertension"
                        else "Monitor and reassess within 3–6 months." if "Stage-1" in bp_l
                        else "Continue lifestyle monitoring." if bp_l == "elevated"
                        else "Within normal range."))

        lines.append(f"• **Cholesterol:** {p['chol']} mg/dl — *{ch_l}*. "
                     + ("Statin therapy evaluation indicated." if ch_l == "high"
                        else "Dietary intervention advised." if ch_l == "borderline-high"
                        else "Maintain heart-healthy diet."))

        lines.append(f"• **Max Heart Rate:** {p['thalach']} bpm — {hr_res:.0f}% of age-predicted maximum. "
                     + ("Chronotropic incompetence possible; cardiology referral warranted." if hr_res < 80
                        else "Excellent cardiac chronotropic response." if hr_res > 95
                        else "Adequate exercise capacity demonstrated."))

        lines.append(f"• **Chest Pain Type:** {cp_str}. "
                     + ("Typical angina is the most predictive symptom for obstructive CAD." if p["cp"] == 0
                        else "Atypical presentation; thorough workup still essential." if p["cp"] == 1
                        else "Non-anginal pain lowers pre-test CAD probability." if p["cp"] == 2
                        else "Asymptomatic presentation — silent ischemia risk warrants further testing."))

        lines.append(f"• **Resting ECG:** {ecg_str}. "
                     + ("ST-T abnormalities on resting ECG increase ischemic risk significantly." if p["restecg"] == 1
                        else "LVH pattern indicates significant pressure/volume overload." if p["restecg"] == 2
                        else "Normal baseline ECG — good baseline reference."))

        if p["exang"] == 1:
            lines.append("• **Exercise-Induced Angina:** Present — strong indicator of flow-limiting coronary disease. Stress imaging study strongly recommended.")
        else:
            lines.append("• **Exercise-Induced Angina:** Absent — reduces probability of hemodynamically significant stenosis.")

        lines.append(f"• **ST Depression (Oldpeak):** {p['oldpeak']} mm | Slope: {slp_str}. "
                     + ("Significant ST depression with downsloping — high specificity for ischemia." if p["oldpeak"] > 2 and p["slope"] == 2
                        else f"Oldpeak of {p['oldpeak']} mm is clinically meaningful; flat slope adds concern." if p["oldpeak"] > 1 and p["slope"] == 1
                        else "Minimal ST changes on stress — reassuring finding."))

        lines.append(f"• **Coronary Vessels (Fluoroscopy):** {p['ca']} major vessel(s) colored. "
                     + ("Multi-vessel disease confirmed — surgical/interventional cardiology input required." if p["ca"] >= 2
                        else "Single-vessel disease suspected — PCI may be appropriate." if p["ca"] == 1
                        else "No fluoroscopic vessel involvement detected."))

        lines.append(f"• **Thalassemia/Perfusion:** {thl_str}. "
                     + ("Fixed defect indicates prior myocardial infarction or scar." if p["thal"] == 1
                        else "Reversible defect is the hallmark of stress-inducible ischemia." if p["thal"] == 2
                        else "Normal perfusion pattern." if p["thal"] == 0
                        else "Perfusion status indeterminate — repeat nuclear study indicated."))

        lines += [
            "",
            "---",
            f"**📊 Estimated 10-Year CVD Risk (Framingham-based):** ~{fram}%",
            "> *This is an educational approximation. Always use validated clinical calculators.*",
        ]

        return "\n".join(lines)

    def get_risk_flags(self) -> list:
        p = self.p
        flags = []
        bp_l, bp_sev = self._bp_level()
        ch_l, ch_sev = self._chol_level()
        hr_res = self._hr_reserve()

        if p["age"] >= 65: flags.append(("Age ≥ 65", "danger"))
        elif p["age"] >= 50: flags.append(("Age 50–64", "warn"))
        if p["sex"] == 1 and p["age"] >= 45: flags.append(("Male ≥ 45", "warn"))
        if p["cp"] == 0: flags.append(("Typical Angina", "danger"))
        elif p["cp"] == 3: flags.append(("Asymptomatic (Silent)", "warn"))
        if p["trestbps"] >= 140: flags.append((f"BP {p['trestbps']} mmHg", "danger"))
        elif p["trestbps"] >= 130: flags.append((f"BP {p['trestbps']} mmHg", "warn"))
        if p["chol"] >= 240: flags.append((f"Chol {p['chol']} mg/dl", "danger"))
        elif p["chol"] >= 200: flags.append((f"Chol {p['chol']} mg/dl", "warn"))
        if p["fbs"] == 1: flags.append(("High Fasting Glucose", "warn"))
        if p["restecg"] == 1: flags.append(("ST-T ECG Abnormality", "danger"))
        elif p["restecg"] == 2: flags.append(("LV Hypertrophy ECG", "danger"))
        if p["thalach"] < 120: flags.append(("Very Low Max HR", "danger"))
        elif hr_res < 85: flags.append(("Reduced HR Reserve", "warn"))
        if p["exang"] == 1: flags.append(("Exercise Angina", "danger"))
        if p["oldpeak"] > 2: flags.append((f"ST ↓ {p['oldpeak']}mm", "danger"))
        elif p["oldpeak"] > 1: flags.append((f"ST ↓ {p['oldpeak']}mm", "warn"))
        if p["slope"] == 2: flags.append(("Downsloping ST", "danger"))
        if p["ca"] >= 2: flags.append((f"{p['ca']} Vessels Diseased", "danger"))
        elif p["ca"] == 1: flags.append(("1 Vessel Diseased", "warn"))
        if p["thal"] == 2: flags.append(("Reversible Defect", "danger"))
        elif p["thal"] == 1: flags.append(("Fixed Defect", "warn"))

        if p["thalach"] >= 160 and p["exang"] == 0: flags.append(("High HR, No Angina", "ok"))
        if p["chol"] < 200: flags.append(("Optimal Cholesterol", "ok"))
        if p["trestbps"] < 120: flags.append(("Normal BP", "ok"))
        if p["ca"] == 0 and p["thal"] == 0: flags.append(("Clean Vessels + Normal Perfusion", "ok"))

        return flags

    def get_recommendations(self) -> list:
        p = self.p
        rec = []
        if self.prob >= 65:
            rec.append("🔴 **Urgent cardiology referral** — arrange within 1–2 weeks")
            rec.append("🔴 Consider **coronary angiography** given multi-factor risk profile")
        elif self.prob >= 40:
            rec.append("🟡 **Cardiology outpatient referral** within 4–6 weeks")
            rec.append("🟡 **Non-invasive stress testing** (exercise ECG or nuclear myocardial perfusion imaging)")
        else:
            rec.append("🟢 **Continue routine cardiovascular surveillance** — annual review")

        if p["trestbps"] >= 140:
            rec.append("💊 Initiate or optimize **antihypertensive therapy** (ACE inhibitor / ARB / CCB)")
        if p["chol"] >= 240:
            rec.append("💊 Initiate **high-intensity statin therapy** (atorvastatin 40–80 mg/day)")
        elif p["chol"] >= 200:
            rec.append("💊 Consider **moderate-intensity statin** + dietary fat modification")
        if p["fbs"] == 1:
            rec.append("🩸 **Diabetes / prediabetes workup** — HbA1c, fasting glucose, OGTT")
        if p["exang"] == 1 or p["oldpeak"] > 2:
            rec.append("🏃 **Supervised cardiac rehab / stress test** before unsupervised exercise")
        else:
            rec.append("🏃 Encourage **150 min/week moderate aerobic activity** (AHA guideline)")
        if p["thal"] == 2:
            rec.append("🔬 **Nuclear perfusion imaging / cardiac MRI** to quantify ischemic territory")
        if p["ca"] >= 2:
            rec.append("🔬 **CT coronary angiography or invasive angiography** for surgical planning")
        if p["sex"] == 0 and p["age"] >= 55:
            rec.append("⚠️ Post-menopausal female: assess **hormone status** & atypical ischemia presentation")
        rec.append("🍎 **Mediterranean diet** adherence — omega-3, fibre, reduce saturated fat")
        rec.append("🚭 Tobacco cessation counselling if applicable")
        rec.append("📋 **Repeat lipid panel + BP** in 3 months to assess treatment response")
        return rec

    def answer_question(self, question: str) -> str:
        q = question.lower().strip()
        p = self.p

        if any(w in q for w in ["hello", "hi ", "hey", "good morning", "good afternoon"]):
            return (f"Hello! I'm Heart Stroke Predictor, your clinical decision support assistant. "
                    f"I've fully analyzed {self.name}'s cardiovascular profile. "
                    f"Current risk classification is **{self.res}** ({self.prob}%). "
                    f"What specific aspect would you like to explore?")

        if any(w in q for w in ["risk", "probability", "chance", "likelihood", "score", "result", "prediction"]):
            tier = "HIGH" if self.prob >= 65 else "MODERATE" if self.prob >= 40 else "LOW"
            bp_l, _ = self._bp_level()
            ch_l, _ = self._chol_level()
            return (f"The predictive model classifies {self.name} as **{self.res}** with a cardiac event probability of **{self.prob}%** — placing them in the **{tier} RISK** tier.\n\n"
                    f"Key drivers of this score include:\n"
                    f"- Chest pain type: {CP_LABELS[p['cp']]}\n"
                    f"- Blood pressure: {p['trestbps']} mmHg ({bp_l})\n"
                    f"- Cholesterol: {p['chol']} mg/dl ({ch_l})\n"
                    f"- Exercise angina: {'Present ⚠️' if p['exang'] else 'Absent ✅'}\n"
                    f"- Vessels involved: {p['ca']}\n"
                    f"- ST oldpeak: {p['oldpeak']} mm\n\n"
                    f"{'Urgent referral is advisable.' if self.prob >= 65 else 'Close monitoring and outpatient cardiology review recommended.' if self.prob >= 40 else 'Continue preventive care with annual monitoring.'}")

        if any(w in q for w in ["blood pressure", "bp", "hypertension", "trestbps", "mmhg", "systolic"]):
            bp_l, bp_sev = self._bp_level()
            resp = (f"**Resting BP: {p['trestbps']} mmHg** — classified as *{bp_l}*.\n\n")
            if p["trestbps"] >= 140:
                resp += ("This is Stage-2 hypertension (≥140 mmHg) per ACC/AHA 2017 guidelines. "
                         "Risks include increased afterload, LV hypertrophy, and accelerated atherosclerosis. "
                         "Immediate antihypertensive therapy is indicated — first-line options include ACE inhibitors, ARBs, thiazide diuretics, or CCBs depending on comorbidities.")
            elif p["trestbps"] >= 130:
                resp += ("Stage-1 hypertension (130–139 mmHg). Lifestyle modification (DASH diet, sodium restriction <1.5g/day, aerobic exercise) should be initiated. "
                         "Pharmacotherapy if 10-year ASCVD risk ≥10%.")
            elif p["trestbps"] >= 120:
                resp += "Elevated BP (120–129 mmHg). Lifestyle modification recommended; no medication needed at this stage unless other risk factors are present."
            else:
                resp += "Normal blood pressure. Continue current lifestyle habits to maintain this."
            return resp

        if any(w in q for w in ["cholesterol", "chol", "lipid", "statin", "ldl", "hdl"]):
            ch_l, _ = self._chol_level()
            resp = f"**Serum Cholesterol: {p['chol']} mg/dl** — *{ch_l}*.\n\n"
            if p["chol"] >= 240:
                resp += ("High cholesterol is a major modifiable CVD risk factor. "
                         "High-intensity statin therapy (e.g., atorvastatin 40–80 mg) is indicated per AHA/ACC guidelines. "
                         "LDL-C target: <70 mg/dl in high-risk patients, <55 mg/dl in very high-risk. "
                         "Consider fasting lipid panel to differentiate LDL/HDL/triglycerides.")
            elif p["chol"] >= 200:
                resp += ("Borderline-high. Dietary intervention: increase soluble fibre (oats, legumes), reduce saturated fat <7% of total calories, avoid trans fats. "
                         "Moderate-intensity statin may be warranted based on overall risk profile.")
            else:
                resp += "Optimal cholesterol level. Maintain Mediterranean-style diet and regular exercise."
            return resp

        if any(w in q for w in ["heart rate", "thalach", "bpm", "pulse", "max hr", "chronotropic"]):
            hr_res = self._hr_reserve()
            pred_max = 220 - p["age"]
            resp = (f"**Max Heart Rate Achieved: {p['thalach']} bpm**\n"
                    f"Age-predicted maximum: {pred_max} bpm\n"
                    f"Heart Rate Reserve utilised: **{hr_res:.0f}%**\n\n")
            if hr_res < 80:
                resp += ("Chronotropic incompetence (failure to achieve ≥80% predicted HR) is associated with increased all-cause mortality and is an independent CVD risk marker. "
                         "This warrants electrophysiology evaluation and may limit stress test interpretability.")
            elif hr_res > 95:
                resp += "Excellent chronotropic response — the patient achieves near-maximum predicted heart rate, indicating good cardiac reserve."
            else:
                resp += "Adequate heart rate response to exercise. Monitoring recommended at next stress test."
            return resp

        if any(w in q for w in ["chest pain", "angina", "cp ", "pain type"]):
            resp = (f"**Chest Pain Classification: {CP_LABELS[p['cp']]}**\n\n"
                    f"The Diamond & Forrester pre-test probability framework categorises chest pain into four types:\n\n"
                    f"- **Typical Angina (Type 0):** All 3 criteria — substernal discomfort, provoked by exertion, relieved by rest/nitrate. Highest pre-test CAD probability.\n"
                    f"- **Atypical Angina (Type 1):** 2 of 3 criteria. Intermediate probability.\n"
                    f"- **Non-Anginal (Type 2):** 1 of 3 criteria. Lower CAD probability.\n"
                    f"- **Asymptomatic (Type 3):** No symptoms — silent ischemia risk.\n\n"
                    f"This patient presents with **{CP_LABELS[p['cp']]}**, "
                    + ("which carries the highest pre-test CAD probability." if p["cp"] == 0
                       else "which is still clinically significant and warrants workup." if p["cp"] == 1
                       else "which is less specific but not to be ignored in context of other risk factors." if p["cp"] == 2
                       else "which may indicate silent ischemia — particularly dangerous as patients may not seek timely care."))
            return resp

        if any(w in q for w in ["ecg", "ekg", "electrocardiogram", "st segment", "restecg", "st-t", "lv hypertrophy", "lvh"]):
            resp = f"**Resting ECG: {ECG_LABELS[p['restecg']]}**\n\n"
            if p["restecg"] == 1:
                resp += ("ST-T wave abnormalities on resting ECG suggest myocardial ischemia, electrolyte imbalance, or medication effect. "
                         "This doubles the risk of future cardiac events compared to a normal baseline ECG. "
                         "Serial ECGs and biomarker assessment (troponin) are recommended.")
            elif p["restecg"] == 2:
                resp += ("Left ventricular hypertrophy (LVH) indicates the heart has been working against increased resistance (hypertension, aortic stenosis). "
                         "LVH increases risk of heart failure, arrhythmia, and sudden cardiac death. "
                         "Echocardiogram is indicated to quantify LV wall thickness and ejection fraction.")
            else:
                resp += "Normal resting ECG. A valuable baseline for comparison during future stress testing."
            return resp

        if any(w in q for w in ["exang", "exercise angina", "exercise induced", "stress test"]):
            if p["exang"] == 1:
                return ("**Exercise-Induced Angina: Present** ⚠️\n\n"
                        "The onset of anginal symptoms during exertion is the hallmark of obstructive coronary artery disease. "
                        "It indicates that myocardial oxygen demand during exercise exceeds supply — confirming flow-limiting stenosis.\n\n"
                        "Clinical pathway:\n"
                        "1. Stop current unsupervised exercise immediately\n"
                        "2. Short-acting nitrate (sublingual GTN) for symptom relief\n"
                        "3. Arrange urgent stress imaging (nuclear MPI or stress echo)\n"
                        "4. Cardiology referral for revascularisation assessment (PCI vs CABG)")
            else:
                return ("**Exercise-Induced Angina: Absent** ✅\n\n"
                        "The absence of exercise-induced angina significantly reduces the probability of hemodynamically significant CAD. "
                        "This is a reassuring finding, though it does not exclude non-obstructive or microvascular disease. "
                        "Regular supervised exercise at 50–75% maximum HR is appropriate and beneficial.")

        if any(w in q for w in ["oldpeak", "st depression", "st slope", "slope", "st change"]):
            resp = (f"**ST Depression (Oldpeak): {p['oldpeak']} mm | Slope: {SLOPE_LBLS[p['slope']]}**\n\n")
            if p["oldpeak"] > 2:
                resp += f"Significant ST depression of {p['oldpeak']} mm is highly specific for myocardial ischemia during exercise. "
            elif p["oldpeak"] > 1:
                resp += f"Moderate ST depression of {p['oldpeak']} mm is borderline significant. "
            else:
                resp += f"Minimal ST depression of {p['oldpeak']} mm — unlikely to represent ischemia alone. "
            resp += ("\n\nST slope interpretation:\n"
                     "- **Upsloping:** Generally benign; may be normal variant in high HR states\n"
                     "- **Flat (horizontal):** Intermediate concern; requires clinical correlation\n"
                     "- **Downsloping:** Most pathological — high specificity for ischemia, associated with worst prognosis\n\n"
                     f"This patient has a **{SLOPE_LBLS[p['slope']]}** ST slope, "
                     + ("which combined with significant depression is a strong ischemic indicator." if p["slope"] == 2 and p["oldpeak"] > 1
                        else "which requires monitoring in the context of other findings." if p["slope"] == 1
                        else "which is reassuring."))
            return resp

        if any(w in q for w in ["vessel", "ca ", " ca", "coronary artery", "fluoroscop", "stenosis", "blockage", "pci", "cabg", "stent"]):
            resp = f"**Major Vessels with Significant Stenosis: {p['ca']}**\n\n"
            if p["ca"] == 0:
                resp += "No major vessels show significant stenosis on fluoroscopy — a highly reassuring finding that substantially lowers the pre-test probability of obstructive CAD."
            elif p["ca"] == 1:
                resp += ("Single-vessel disease identified. This is typically amenable to **percutaneous coronary intervention (PCI)** with stenting. "
                         "The LAD territory should be evaluated first given prognostic impact. "
                         "Fractional flow reserve (FFR) guidance improves PCI outcomes.")
            elif p["ca"] == 2:
                resp += ("Two-vessel disease present. Treatment decision depends on SYNTAX score, LV function, and diabetic status. "
                         "Both PCI and CABG are viable — a Heart Team discussion is recommended per ESC guidelines.")
            else:
                resp += ("Three-vessel or left main disease. **CABG is typically the preferred revascularisation strategy** based on survival benefit demonstrated in SYNTAX, FREEDOM, and EXCEL trials. "
                         "Urgent surgical cardiology consultation is indicated.")
            return resp

        if any(w in q for w in ["thal", "perfusion", "defect", "nuclear", "myocardial", "scar", "viability"]):
            resp = f"**Myocardial Perfusion (Thalassemia): {THAL_LABELS[p['thal']]}**\n\n"
            if p["thal"] == 0:
                resp += "Normal perfusion pattern — no fixed or inducible defects. Good prognostic sign."
            elif p["thal"] == 1:
                resp += ("**Fixed perfusion defect** indicates transmural myocardial scar — prior MI with no viable myocardium in that territory. "
                         "This region will not benefit from revascularisation. Cardiac MRI with gadolinium can delineate scar extent and guide decisions about ICD therapy.")
            elif p["thal"] == 2:
                resp += ("**Reversible perfusion defect** is the gold standard finding of stress-inducible ischemia. "
                         "The territory shows reduced uptake during stress but normalises at rest — confirming viable but hibernating myocardium. "
                         "This is the primary indication for revascularisation to restore function and improve prognosis.")
            else:
                resp += "Perfusion status is indeterminate. Repeat nuclear imaging with optimal patient preparation is recommended."
            return resp

        if any(w in q for w in ["fbs", "diabetes", "blood sugar", "glucose", "hba1c", "fasting"]):
            if p["fbs"] == 1:
                return ("**Fasting Blood Sugar > 120 mg/dl** ⚠️\n\n"
                        "Elevated fasting glucose suggests prediabetes or undiagnosed type 2 diabetes. "
                        "Diabetes is an independent and potent cardiovascular risk multiplier — it accelerates atherosclerosis, promotes microvascular disease, and worsens outcomes after MI.\n\n"
                        "Recommended workup:\n"
                        "- HbA1c (target <7.0% in diabetics, <6.5% in select patients)\n"
                        "- Fasting lipid panel (diabetic dyslipidaemia: ↑TG, ↓HDL)\n"
                        "- Urine albumin-creatinine ratio (early nephropathy marker)\n"
                        "- Refer to endocrinology / diabetes nurse educator\n\n"
                        "SGLT2 inhibitors (e.g., empagliflozin) and GLP-1 agonists (e.g., liraglutide) offer additional CV outcome benefits in diabetics.")
            else:
                return ("**Fasting Blood Sugar ≤ 120 mg/dl** ✅\n\n"
                        "Normal fasting glucose. This removes diabetes as an active driver of cardiovascular risk in this assessment. "
                        "Continue healthy dietary habits to maintain glycaemic control.")

        if any(w in q for w in ["age", "sex", "gender", "male", "female", "woman", "man"]):
            resp = f"**Demographics: {SEX_LABELS[p['sex']]}, Age {p['age']}**\n\n"
            if p["sex"] == 1 and p["age"] >= 45:
                resp += ("Males ≥45 years carry significantly elevated baseline cardiovascular risk. "
                         "Testosterone decline and associated metabolic changes (central obesity, insulin resistance) accelerate atherosclerosis. "
                         "ACC/AHA 2019 guidelines recommend calculating 10-year ASCVD risk for statin initiation decisions.")
            elif p["sex"] == 0 and p["age"] >= 55:
                resp += ("Post-menopausal females (typically ≥55) lose the cardioprotective effects of endogenous oestrogen. "
                         "CAD presentation in women is often atypical (fatigue, jaw pain, nausea) — leading to underdiagnosis. "
                         "Microvascular disease (coronary microvascular dysfunction) is more prevalent in women and may require different diagnostic approaches.")
            elif p["age"] < 45:
                resp += ("Premature cardiovascular disease in patients <45 warrants investigation for inherited conditions: "
                         "familial hypercholesterolaemia (FH), hypertrophic cardiomyopathy (HCM), or primary hypercoagulable states. "
                         "Genetic counselling may be appropriate.")
            return resp

        if any(w in q for w in ["medication", "drug", "medicine", "treatment", "therapy", "prescri"]):
            meds = ["🫀 **Aspirin 75–100 mg/day** — antiplatelet therapy for CAD risk reduction"]
            if p["trestbps"] >= 130:
                meds.append("💊 **ACE inhibitor** (e.g., ramipril 5–10 mg) or **ARB** for BP and cardioprotection")
            if p["chol"] >= 200:
                meds.append("💊 **Statin** — atorvastatin 40–80 mg (high-intensity) if high risk, 10–20 mg if moderate risk")
            if p["exang"] == 1 or p["oldpeak"] > 1.5:
                meds.append("💊 **Beta-blocker** (e.g., bisoprolol 2.5–10 mg) for angina symptom control and mortality benefit")
                meds.append("💊 **Sublingual GTN** spray for acute anginal episodes")
            if p["fbs"] == 1:
                meds.append("🩸 **Metformin** (first-line diabetes) ± SGLT2 inhibitor for CV outcome benefit")
            meds.append("🐟 **Omega-3 fatty acids** 1–4g/day if triglycerides elevated")
            return ("Pharmacological considerations for this patient:\n\n" + "\n".join(meds) +
                    "\n\n> ⚠️ *All prescribing decisions must be made by a licensed clinician with full patient assessment.*")

        if any(w in q for w in ["lifestyle", "diet", "exercise", "weight", "smoking", "alcohol", "stress", "sleep"]):
            return ("**Lifestyle Modification Plan:**\n\n"
                    "🍎 **Diet:**\n"
                    "- Mediterranean diet: olive oil, fish ≥2×/week, legumes, whole grains, vegetables\n"
                    "- Reduce sodium to <1.5g/day (for hypertension)\n"
                    "- Limit saturated fat <7% total calories; eliminate trans fats\n"
                    "- Increase dietary fibre to 25–38g/day\n\n"
                    "🏃 **Physical Activity:**\n"
                    "- 150 min/week moderate aerobic exercise OR 75 min vigorous (AHA)\n"
                    "- Resistance training 2×/week\n"
                    + ("- **Start with supervised cardiac rehab** before independent exercise given current risk profile\n" if self.prob >= 40 else "- Walking, cycling, swimming are excellent low-impact options\n")
                    + "\n🚭 **Tobacco:**\n- Cessation reduces CV risk by 50% within 1 year. NRT, varenicline (Champix), or bupropion\n\n"
                    "🍺 **Alcohol:**\n- Limit to <14 units/week (men), <7 units/week (women)\n\n"
                    "😴 **Sleep & Stress:**\n- Target 7–9 hours/night; treat OSA if present (CPAP)\n"
                    "- Mindfulness-based stress reduction (MBSR) has demonstrable CV benefit")

        if any(w in q for w in ["prognosis", "outlook", "future", "survival", "long term", "lifespan", "5 year", "10 year"]):
            fram = self._framingham_risk_estimate()
            return (f"**Prognostic Assessment for {self.name}:**\n\n"
                    f"- Model-predicted cardiac event probability: **{self.prob}%**\n"
                    f"- Estimated 10-year CVD risk (Framingham approximation): **~{fram}%**\n"
                    f"- Risk classification: **{self.res}**\n\n"
                    + ("⚠️ **High-risk prognosis.** Without intervention, high-risk patients face significantly elevated rates of acute coronary syndrome, heart failure, and cardiovascular mortality. "
                       "Aggressive risk factor modification and revascularisation (if indicated) can substantially improve outcomes.\n\n"
                       "Evidence-based interventions that improve survival:\n"
                       "- Statins: 25–35% RRR in major cardiovascular events\n"
                       "- ACE inhibitors post-MI: 20% mortality reduction\n"
                       "- Beta-blockers: 23% mortality reduction in CAD\n"
                       "- SGLT2 inhibitors in HF: 25% HHF/CV death reduction"
                       if self.prob >= 65 else
                       "🟡 **Moderate-risk prognosis.** With appropriate intervention and risk factor modification, outcomes can be significantly improved. "
                       "Non-invasive workup to rule out significant ischemia is the priority next step."
                       if self.prob >= 40 else
                       "🟢 **Favourable prognosis.** Low-risk profile suggests good long-term cardiovascular outlook. "
                       "Maintaining current lifestyle with annual monitoring is the recommended approach."))

        if any(w in q for w in ["refer", "specialist", "cardiologist", "hospital", "admit", "emergency"]):
            if self.prob >= 65:
                return ("**Referral Recommendation: URGENT** 🔴\n\n"
                        "Given the high-risk prediction, the following referral pathway is recommended:\n\n"
                        "1. **Same-day cardiology consultation** if patient is symptomatic at rest or has new ECG changes\n"
                        "2. **Elective cardiology referral (within 2 weeks)** if currently stable\n"
                        "3. Investigations to arrange before referral:\n"
                        "   - 12-lead ECG\n"
                        "   - Troponin I/T (high-sensitivity)\n"
                        "   - Full blood count, U&E, LFTs, thyroid function\n"
                        "   - Fasting lipid panel + HbA1c\n"
                        "   - Chest X-ray\n"
                        "   - Echocardiogram\n"
                        "4. If unstable: **emergency department attendance / 999 / ambulance**")
            elif self.prob >= 40:
                return ("**Referral Recommendation: ROUTINE OUTPATIENT** 🟡\n\n"
                        "- Outpatient cardiology referral within 4–6 weeks\n"
                        "- Pre-referral workup: ECG, fasting bloods, echocardiogram if BP elevated\n"
                        "- Consider 24hr Holter if palpitations or arrhythmia suspected\n"
                        "- Exercise treadmill test (ETT) or nuclear MPI if stress testing indicated")
            else:
                return ("**Referral Recommendation: NOT IMMEDIATELY REQUIRED** 🟢\n\n"
                        "- Manage in primary care with annual cardiovascular risk review\n"
                        "- Refer to dietitian / lifestyle medicine if needed\n"
                        "- Consider cardiology referral only if symptoms develop or risk factors worsen")

        if any(w in q for w in ["summary", "overview", "report", "overall", "tell me about", "explain"]):
            return self.generate_summary()

        if any(w in q for w in ["what should", "next step", "plan", "action", "do now", "recommend"]):
            recs = self.get_recommendations()
            return "**Recommended Clinical Action Plan:**\n\n" + "\n".join(recs)

        if any(w in q for w in ["normal", "compare", "typical", "average", "benchmark", "range"]):
            return (f"**Clinical Reference Ranges vs {self.name}'s Values:**\n\n"
                    f"| Parameter | Patient Value | Normal Range | Status |\n"
                    f"|-----------|-------------|--------------|--------|\n"
                    f"| Blood Pressure | {p['trestbps']} mmHg | <120 mmHg | {'🔴 High' if p['trestbps'] >= 140 else '🟡 Elevated' if p['trestbps'] >= 120 else '🟢 Normal'} |\n"
                    f"| Cholesterol | {p['chol']} mg/dl | <200 mg/dl | {'🔴 High' if p['chol'] >= 240 else '🟡 Borderline' if p['chol'] >= 200 else '🟢 Normal'} |\n"
                    f"| Max Heart Rate | {p['thalach']} bpm | {int(220-p['age']*0.85)}–{220-p['age']} bpm | {'🔴 Low' if p['thalach'] < 120 else '🟡 Below target' if p['thalach'] < int(220-p['age']*0.85) else '🟢 Normal'} |\n"
                    f"| Fasting Glucose | {'High (>120)' if p['fbs'] else 'Normal (≤120)'} mg/dl | ≤100 mg/dl | {'🔴 Elevated' if p['fbs'] else '🟢 Normal'} |\n"
                    f"| ST Oldpeak | {p['oldpeak']} mm | <1 mm | {'🔴 Significant' if p['oldpeak'] > 2 else '🟡 Borderline' if p['oldpeak'] > 1 else '🟢 Normal'} |")

        if any(w in q for w in ["framingham", "ascvd", "10 year", "10-year", "cvd risk"]):
            fram = self._framingham_risk_estimate()
            return (f"**Framingham 10-Year CVD Risk Estimate: ~{fram}%**\n\n"
                    f"This is an educational approximation based on: age ({p['age']}), sex ({SEX_LABELS[p['sex']]}), "
                    f"BP ({p['trestbps']} mmHg), cholesterol ({p['chol']} mg/dl), diabetes status, and exercise angina.\n\n"
                    f"Risk stratification:\n"
                    f"- <7.5%: Low risk\n"
                    f"- 7.5–10%: Borderline risk\n"
                    f"- 10–20%: Intermediate risk\n"
                    f"- >20%: High risk\n\n"
                    f"For validated calculation, use the ACC/AHA ASCVD Risk Estimator Plus at tools.acc.org/ASCVD-Risk-Estimator-Plus/"
                    f"\n\n> ⚠️ This is an approximation only. Always use validated clinical tools for decision-making.")

        return (f"I can provide detailed analysis on any of these clinical aspects for {self.name}:\n\n"
                "• **Risk & Probability** — model output interpretation\n"
                "• **Blood Pressure** — hypertension staging & treatment\n"
                "• **Cholesterol** — lipid management\n"
                "• **Heart Rate** — chronotropic assessment\n"
                "• **Chest Pain** — Diamond-Forrester classification\n"
                "• **ECG findings** — ST changes & LVH\n"
                "• **Exercise Angina** — ischemia confirmation\n"
                "• **ST Oldpeak & Slope** — ischemic thresholds\n"
                "• **Coronary Vessels** — disease burden & revascularisation\n"
                "• **Myocardial Perfusion** — defect classification\n"
                "• **Medications** — pharmacological options\n"
                "• **Lifestyle** — evidence-based modification\n"
                "• **Prognosis** — long-term outlook\n"
                "• **Referral** — clinical pathway\n"
                "• **Framingham Risk** — 10-year CVD estimate\n"
                "• **Normal Ranges** — comparative benchmarks\n\n"
                "What would you like to explore?")


# ─────────────────────────────────────────────────────────────────────────────
# PDF REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf_report(patient_name, pd_, result, prob, risk_flags, recommendations, summary_text):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()

    def make_style(name, parent="Normal", **kwargs):
        return ParagraphStyle(name, parent=styles[parent], **kwargs)

    title_style  = make_style("Title2",  parent="Title",   fontSize=22, textColor=colors.HexColor("#002B36"), leading=28)
    head2_style  = make_style("Head2",   parent="Heading2", fontSize=13, textColor=colors.HexColor("#00626B"), spaceAfter=4)
    body_style   = make_style("Body2",   fontSize=9.5,     leading=14,  textColor=colors.HexColor("#1A2733"))
    label_style  = make_style("Label",   fontSize=8,       textColor=colors.HexColor("#4A6080"), spaceAfter=2)
    risk_high    = make_style("RiskH",   fontSize=12,      textColor=colors.HexColor("#B01535"), fontName="Helvetica-Bold")
    risk_low     = make_style("RiskL",   fontSize=12,      textColor=colors.HexColor("#047857"), fontName="Helvetica-Bold")
    warn_style   = make_style("Warn",    fontSize=8,       textColor=colors.HexColor("#666666"), leading=11)

    story = []

    story.append(Paragraph("Heart Stroke Predictor", title_style))
    story.append(Paragraph("Clinical Heart Stroke Risk Assessment Report", label_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#00626B")))
    story.append(Spacer(1, 0.15*inch))

    story.append(Paragraph("Patient Information", head2_style))
    patient_data_table = [
        ["Patient Name", patient_name, "Assessment Date", datetime.now().strftime("%B %d, %Y")],
        ["Age", str(pd_["age"]), "Sex", "Male" if pd_["sex"] == 1 else "Female"],
        ["Model Prediction", result, "Risk Probability", f"{prob}%"],
    ]
    t = Table(patient_data_table, colWidths=[1.3*inch, 2*inch, 1.3*inch, 2*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, -1), colors.HexColor("#F4FAFB")),
        ("TEXTCOLOR",      (0, 0), (0, -1),  colors.HexColor("#4A6080")),
        ("TEXTCOLOR",      (2, 0), (2, -1),  colors.HexColor("#4A6080")),
        ("FONTNAME",       (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME",       (0, 0), (0, -1),  "Helvetica-Bold"),
        ("FONTNAME",       (2, 0), (2, -1),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#EAF5F7"), colors.HexColor("#F4FAFB")]),
        ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#C8DDE0")),
        ("PADDING",        (0, 0), (-1, -1), 7),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.15*inch))

    risk_sty = risk_high if result == "High Risk" else risk_low
    story.append(Paragraph(f"Risk Classification: {result} ({prob}%)", risk_sty))
    story.append(Spacer(1, 0.1*inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#C8DDE0")))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph("Clinical Parameters", head2_style))
    params = [
        ["Parameter", "Value", "Reference", "Status"],
        ["Resting BP",    f"{pd_['trestbps']} mmHg", "< 120 mmHg", "High" if pd_['trestbps'] >= 130 else "OK"],
        ["Cholesterol",   f"{pd_['chol']} mg/dl",    "< 200 mg/dl", "High" if pd_['chol'] >= 200 else "OK"],
        ["Max Heart Rate",f"{pd_['thalach']} bpm",   f"{int(0.85*(220-pd_['age']))}–{220-pd_['age']} bpm",
                          "OK" if pd_['thalach'] >= int(0.85*(220-pd_['age'])) else "Low"],
        ["Fasting Glucose", "> 120 mg/dl" if pd_['fbs'] else "<= 120 mg/dl", "<= 100 mg/dl",
                          "Elevated" if pd_['fbs'] else "OK"],
        ["ST Oldpeak",    f"{pd_['oldpeak']} mm",    "< 1 mm",      "Abnormal" if pd_['oldpeak'] > 1 else "OK"],
        ["Chest Pain Type", CP_LABELS[pd_['cp']],    "Non-Anginal", "Angina" if pd_['cp'] <= 1 else "Assess"],
        ["Resting ECG",   ECG_LABELS[pd_['restecg']],"Normal",      "Abnormal" if pd_['restecg'] > 0 else "Normal"],
        ["Exercise Angina","Yes" if pd_['exang'] else "No", "No",   "Present" if pd_['exang'] else "Absent"],
        ["Coronary Vessels", str(pd_['ca']),          "0",           "Diseased" if pd_['ca'] > 0 else "Clean"],
        ["Thal/Perfusion", THAL_LABELS[pd_['thal']], "Normal",      "Defect" if pd_['thal'] in [1, 2] else "Normal"],
    ]
    tbl = Table(params, colWidths=[1.6*inch, 1.4*inch, 1.4*inch, 1.4*inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  colors.HexColor("#00626B")),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#EAF5F7"), colors.HexColor("#F4FAFB")]),
        ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#C8DDE0")),
        ("PADDING",        (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.15*inch))

    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#C8DDE0")))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("Clinical Recommendations", head2_style))
    for rec in recommendations:
        clean = re.sub(r'[^\x00-\x7F\u00C0-\u024F\u2013\u2014\u2018\u2019\u201C\u201D]', '', rec)
        clean = clean.replace("**", "").strip()
        if clean:
            story.append(Paragraph(f"• {clean}", body_style))
            story.append(Spacer(1, 0.04*inch))

    story.append(Spacer(1, 0.1*inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#C8DDE0")))
    story.append(Spacer(1, 0.08*inch))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI-assisted clinical decision support system and is intended "
        "for educational and supplementary purposes only. All clinical decisions must be made by qualified healthcare "
        "professionals based on comprehensive patient assessment. This tool does not replace clinical judgement, "
        "physical examination, or validated diagnostic protocols.",
        warn_style))

    doc.build(story)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🫀 Heart Stroke Predictor")
    st.markdown("<p style='color:#4A6080;font-size:0.78rem;margin-top:-0.5rem;font-family:IBM Plex Mono;'>Clinical Decision Support v2</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-header">Patient Profile</div>', unsafe_allow_html=True)
    name     = st.text_input("Patient Name", "Anonymous")
    age      = st.slider("Age", 18, 100, 52)
    sex      = st.selectbox("Biological Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

    st.markdown('<div class="section-header" style="margin-top:1rem;">Clinical Measurements</div>', unsafe_allow_html=True)
    trestbps = st.number_input("Resting Blood Pressure (mmHg)", 80, 200, 130)
    chol     = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 250)
    fbs      = st.selectbox("Fasting Blood Sugar", [0, 1],
                             format_func=lambda x: "<= 120 mg/dl (Normal)" if x == 0 else "> 120 mg/dl (High)")
    thalach  = st.slider("Maximum Heart Rate Achieved", 60, 220, 158)
    oldpeak  = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, 0.1)

    st.markdown('<div class="section-header" style="margin-top:1rem;">Diagnostic Features</div>', unsafe_allow_html=True)
    cp       = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                             format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina",
                                                    2: "Non-Anginal", 3: "Asymptomatic"}[x])
    restecg  = st.selectbox("Resting ECG", [0, 1, 2],
                             format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}[x])
    exang    = st.selectbox("Exercise Induced Angina", [0, 1],
                             format_func=lambda x: "No" if x == 0 else "Yes")
    slope    = st.selectbox("ST Slope", [0, 1, 2],
                             format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x])
    ca       = st.selectbox("Major Vessels (Fluoroscopy)", [0, 1, 2, 3])
    thal     = st.selectbox("Thalassemia", [0, 1, 2, 3],
                             format_func=lambda x: {0: "Normal", 1: "Fixed Defect",
                                                    2: "Reversible Defect", 3: "Unknown"}[x])

    st.markdown("---")
    predict_btn = st.button("⚡ Run Prediction", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────────────────────────────────────
patient = dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol,
               fbs=fbs, restecg=restecg, thalach=thalach, exang=exang,
               oldpeak=oldpeak, slope=slope, ca=ca, thal=thal)

if predict_btn:
    input_df    = pd.DataFrame([patient], columns=columns)
    scaled      = scaler.transform(input_df)
    pred        = model.predict(scaled)[0]
    proba       = model.predict_proba(scaled)[0]
    probability = int(proba[1] * 100)
    result      = "High Risk" if pred == 1 else "Low Risk"

    engine = ClinicalEngine(patient, name, result, probability)

    st.session_state["result"]          = result
    st.session_state["probability"]     = probability
    st.session_state["patient_data"]    = patient
    st.session_state["patient_name"]    = name
    st.session_state["prediction_done"] = True
    st.session_state["risk_flags"]      = engine.get_risk_flags()
    st.session_state["recommendations"] = engine.get_recommendations()
    st.session_state["clinical_summary"]= engine.generate_summary()
    st.session_state["chat_history"]    = []

# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────
col_logo, col_title, col_status = st.columns([0.5, 3, 2])
with col_logo:
    st.markdown("<h1 style='font-size:2.2rem;margin:0;'>🫀</h1>", unsafe_allow_html=True)
with col_title:
    st.markdown(
        "<h1 style='margin:0;font-size:1.9rem;background:linear-gradient(90deg,#00E5C3,#67ECD6,#38BDF8);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-family:Syne,sans-serif;"
        "font-weight:800;'>Heart Stroke Predictor</h1>",
        unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#4A6080;margin:0;font-size:0.72rem;letter-spacing:0.15em;font-family:IBM Plex Mono;'>"
        "CLINICAL HEART STROKE RISK ASSESSMENT SYSTEM</p>",
        unsafe_allow_html=True)
with col_status:
    if st.session_state["prediction_done"]:
        res   = st.session_state["result"]
        prob  = st.session_state["probability"]
        badge = "badge-high" if res == "High Risk" else "badge-low"
        icon  = "⚠️" if res == "High Risk" else "✅"
        st.markdown(
            f"<div style='text-align:right;padding-top:0.4rem;'>"
            f"<span class='{badge}'>{icon} {res} · {prob}%</span></div>",
            unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_dashboard, tab_flags, tab_ai, tab_report = st.tabs(
    ["📊  Dashboard", "🚦  Risk Flags", "🤖  AI Assistant", "📄  Report"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    if not st.session_state["prediction_done"]:
        st.markdown("""
        <div style='text-align:center;padding:4rem 2rem;'>
            <div style='font-size:4rem;'>🫀</div>
            <h2 style='color:#00E5C3;font-family:Syne,sans-serif;'>Enter Patient Data &amp; Run Prediction</h2>
            <p style='color:#4A6080;'>Complete the patient profile in the sidebar and click <b>Run Prediction</b>.</p>
        </div>""", unsafe_allow_html=True)
    else:
        prob  = st.session_state["probability"]
        res   = st.session_state["result"]
        pd_   = st.session_state["patient_data"]
        pname = st.session_state["patient_name"]

        def kpi(col, label, value, color="#00E5C3"):
            col.markdown(
                f"<div class='metric-tile'>"
                f"<div class='metric-value' style='color:{color};'>{value}</div>"
                f"<div class='metric-label'>{label}</div></div>",
                unsafe_allow_html=True)

        k1, k2, k3, k4, k5 = st.columns(5)
        kpi(k1, "Risk Score",   f"{prob}%",
            "#FF4D6D" if prob > 65 else "#F59E0B" if prob > 40 else "#10B981")
        kpi(k2, "Max HR",       f"{pd_['thalach']} bpm", "#00E5C3")
        kpi(k3, "Cholesterol",  f"{pd_['chol']} mg/dl",
            "#F59E0B" if pd_['chol'] > 240 else "#00E5C3")
        kpi(k4, "Blood Press.", f"{pd_['trestbps']} mmHg",
            "#FF4D6D" if pd_['trestbps'] > 140 else "#F59E0B" if pd_['trestbps'] > 130 else "#00E5C3")
        kpi(k5, "ST Oldpeak",   f"{pd_['oldpeak']} mm",
            "#FF4D6D" if pd_['oldpeak'] > 2 else "#F59E0B" if pd_['oldpeak'] > 1 else "#00E5C3")

        st.markdown("<br>", unsafe_allow_html=True)
        left, right = st.columns([1.2, 1])

        with left:
            st.markdown('<div class="cardio-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Risk Probability Gauge</div>', unsafe_allow_html=True)

            fig_g, ax_g = plt.subplots(figsize=(5, 2.9), facecolor="#081525")
            ax_g.set_facecolor("#081525")
            theta = np.linspace(np.pi, 0, 300)
            ax_g.plot(np.cos(theta), np.sin(theta), color="#0F2840", linewidth=20, solid_capstyle="round")
            zone_colors = ["#047857", "#10B981", "#F59E0B", "#EF4444", "#B01535"]
            for i, zc in enumerate(zone_colors):
                t_s = np.pi - (i / 5) * np.pi
                t_e = np.pi - ((i + 1) / 5) * np.pi
                t_z = np.linspace(t_s, t_e, 60)
                ax_g.plot(np.cos(t_z), np.sin(t_z), color=zc, linewidth=20,
                          alpha=0.45, solid_capstyle="butt")
            active_t = np.linspace(np.pi, np.pi - (prob / 100) * np.pi, 300)
            ncol = "#FF4D6D" if prob > 65 else "#F59E0B" if prob > 40 else "#10B981"
            ax_g.plot(np.cos(active_t), np.sin(active_t), color=ncol,
                      linewidth=20, solid_capstyle="round", alpha=0.95)
            nangle = np.pi - (prob / 100) * np.pi
            ax_g.annotate("",
                          xy=(0.84 * np.cos(nangle), 0.84 * np.sin(nangle)),
                          xytext=(0, 0),
                          arrowprops=dict(arrowstyle="-|>", color=ncol, lw=2.5))
            circle = plt.Circle((0, 0), 0.07, color=ncol, zorder=5)
            ax_g.add_patch(circle)
            ax_g.text(0, 0.18, f"{prob}%", ha="center", va="center",
                      fontsize=24, fontweight="bold", color=ncol, fontfamily="monospace")
            ax_g.text(0, -0.07, res.upper(), ha="center", va="center",
                      fontsize=8, color="#4A6080", fontfamily="monospace")
            ax_g.set_xlim(-1.15, 1.15)
            ax_g.set_ylim(-0.25, 1.15)
            ax_g.axis("off")
            plt.tight_layout(pad=0)
            st.pyplot(fig_g, use_container_width=True)
            plt.close(fig_g)

            engine_tmp = ClinicalEngine(pd_, pname, res, prob)
            fram = engine_tmp._framingham_risk_estimate()
            st.markdown(
                f"<div style='text-align:center;margin-top:0.3rem;'>"
                f"<span style='font-family:IBM Plex Mono;font-size:0.75rem;color:#4A6080;'>"
                f"Est. 10-yr Framingham CVD Risk: </span>"
                f"<span style='font-family:IBM Plex Mono;font-size:0.85rem;color:{ncol};"
                f"font-weight:600;'>~{fram}%</span></div>",
                unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="cardio-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Feature Risk Contribution</div>', unsafe_allow_html=True)

            features_impact = {
                "Age":            min(abs(pd_["age"] - 40) / 60, 1.0) * (1 if pd_["age"] > 55 else 0.4),
                "Chest Pain":     pd_["cp"] / 3 if pd_["cp"] <= 1 else (3 - pd_["cp"]) / 3 * 0.3,
                "Max HR":         min((220 - pd_["thalach"]) / 160, 1.0),
                "ST Oldpeak":     pd_["oldpeak"] / 6,
                "Vessels (CA)":   pd_["ca"] / 3,
                "Cholesterol":    min((pd_["chol"] - 150) / 450, 1.0),
                "Blood Pressure": min((pd_["trestbps"] - 80) / 120, 1.0),
                "Exercise Angina":0.9 if pd_["exang"] else 0.05,
            }
            is_risk = {
                "Age":             pd_["age"] > 55,
                "Chest Pain":      pd_["cp"] <= 1,
                "Max HR":          pd_["thalach"] < 140,
                "ST Oldpeak":      pd_["oldpeak"] > 1,
                "Vessels (CA)":    pd_["ca"] > 0,
                "Cholesterol":     pd_["chol"] > 200,
                "Blood Pressure":  pd_["trestbps"] > 130,
                "Exercise Angina": pd_["exang"] == 1,
            }

            fig_f, ax_f = plt.subplots(figsize=(5, 3.8), facecolor="#081525")
            ax_f.set_facecolor("#081525")
            items   = sorted(features_impact.items(), key=lambda x: x[1], reverse=True)[:7]
            names   = [i[0] for i in items]
            vals    = [i[1] * (1 if is_risk.get(i[0], True) else -0.35) for i in items]
            bcolors = ["#FF4D6D" if v > 0 else "#00E5C3" for v in vals]
            ax_f.barh(names, vals, color=bcolors, height=0.55, edgecolor="none")
            ax_f.axvline(0, color="#0F2840", linewidth=1)
            ax_f.tick_params(colors="#4A6080", labelsize=8)
            ax_f.spines[:].set_visible(False)
            ax_f.set_xlabel("Risk Impact →", color="#4A6080", fontsize=8)
            plt.tight_layout(pad=0.5)
            st.pyplot(fig_f, use_container_width=True)
            plt.close(fig_f)

            st.markdown(
                "<div style='font-size:0.7rem;color:#4A6080;display:flex;gap:1.5rem;'>"
                "<span><span style='color:#FF4D6D;'>■</span> Increases risk</span>"
                "<span><span style='color:#00E5C3;'>■</span> Reduces risk</span></div>",
                unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="cardio-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Clinical Parameters at a Glance</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        vitals = [
            (c1, pd_["trestbps"], "Blood Pressure", "mmHg", 120, 140, 80,  200),
            (c2, pd_["chol"],     "Cholesterol",    "mg/dl",200, 240, 100, 600),
            (c3, pd_["thalach"],  "Max Heart Rate", "bpm",  150, 190, 60,  220),
            (c4, pd_["age"],      "Patient Age",    "yrs",  55,  65,  18,  100),
        ]
        for col_w, val, lbl, unit, warn_v, danger_v, mn, mx in vitals:
            color_v = "#FF4D6D" if val >= danger_v else "#F59E0B" if val >= warn_v else "#10B981"
            pct = min((val - mn) / (mx - mn) * 100, 100)
            col_w.markdown(
                f"<div class='metric-tile'>"
                f"<div class='metric-value' style='color:{color_v};'>{val}"
                f"<span style='font-size:0.7rem;color:#4A6080;'> {unit}</span></div>"
                f"<div class='metric-label'>{lbl}</div>"
                f"<div style='background:#0F2840;border-radius:4px;height:4px;margin-top:0.5rem;'>"
                f"<div style='background:{color_v};width:{pct:.0f}%;height:4px;border-radius:4px;'></div>"
                f"</div></div>",
                unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — RISK FLAGS
# ═══════════════════════════════════════════════════════════════════════════
with tab_flags:
    if not st.session_state["prediction_done"]:
        st.info("🫀 Run a prediction to view risk flags.")
    else:
        st.markdown('<div class="cardio-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Active Risk Flags</div>', unsafe_allow_html=True)

        flags = st.session_state["risk_flags"]
        danger_flags = [(f, s) for f, s in flags if s == "danger"]
        warn_flags   = [(f, s) for f, s in flags if s == "warn"]
        ok_flags     = [(f, s) for f, s in flags if s == "ok"]

        if danger_flags:
            st.markdown("**🔴 High-Risk Indicators**")
            html = "".join([f"<span class='insight-pill pill-danger'>⚠ {f}</span>" for f, _ in danger_flags])
            st.markdown(html, unsafe_allow_html=True)
            st.markdown("")

        if warn_flags:
            st.markdown("**🟡 Moderate Concerns**")
            html = "".join([f"<span class='insight-pill pill-warn'>→ {f}</span>" for f, _ in warn_flags])
            st.markdown(html, unsafe_allow_html=True)
            st.markdown("")

        if ok_flags:
            st.markdown("**🟢 Protective / Normal Findings**")
            html = "".join([f"<span class='insight-pill pill-ok'>✓ {f}</span>" for f, _ in ok_flags])
            st.markdown(html, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="cardio-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Clinical Recommendations</div>', unsafe_allow_html=True)
        for rec in st.session_state["recommendations"]:
            st.markdown(
                f"<p style='margin:0.3rem 0;font-size:0.9rem;line-height:1.5;'>{rec}</p>",
                unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="cardio-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Full Clinical Summary</div>', unsafe_allow_html=True)
        st.markdown(st.session_state["clinical_summary"])
        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — AI ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════
with tab_ai:
    if not st.session_state["prediction_done"]:
        st.info("🫀 Please run a prediction first to enable the AI assistant.")
    else:
        res   = st.session_state["result"]
        prob  = st.session_state["probability"]
        pd_   = st.session_state["patient_data"]
        pname = st.session_state["patient_name"]

        ai_left, ai_right = st.columns([2.2, 1])

        with ai_left:
            st.markdown('<div class="cardio-card" style="min-height:460px;">', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-header">AI Clinical Assistant — Rule-Based Expert System</div>',
                unsafe_allow_html=True)

            # ── Initial greeting with avatar ───────────────────────────────
            if not st.session_state["chat_history"]:
                tier    = "HIGH" if prob >= 65 else "MODERATE" if prob >= 40 else "LOW"
                color_t = "#FF4D6D" if prob >= 65 else "#F59E0B" if prob >= 40 else "#10B981"
                engine  = ClinicalEngine(pd_, pname, res, prob)
                flags   = engine.get_risk_flags()
                danger_count = sum(1 for _, s in flags if s == "danger")
                greeting_html = (
                    f"Hello! I've completed a full clinical analysis of <b>{pname}</b>'s cardiovascular profile.<br><br>"
                    f"<b>Classification:</b> <span style='color:{color_t};font-weight:700;'>{res} ({prob}%)</span> — {tier} RISK TIER<br>"
                    f"<b>Active Risk Flags:</b> {danger_count} high-risk indicators detected<br><br>"
                    f"I can answer detailed clinical questions about any parameter — blood pressure, cholesterol, "
                    f"ECG findings, coronary vessels, medications, lifestyle, prognosis, referral pathways, and more.<br><br>"
                    f"<i style='color:#4A6080;font-size:0.8rem;'>Try: \"What does the ST oldpeak mean?\" or "
                    f"\"What medications are recommended?\" or \"What's the prognosis?\"</i>"
                )
                st.markdown(ai_bubble(greeting_html), unsafe_allow_html=True)

            # ── Chat history with avatars ──────────────────────────────────
            for msg in st.session_state["chat_history"]:
                if msg["role"] == "user":
                    st.markdown(user_bubble(msg["content"]), unsafe_allow_html=True)
                else:
                    # Convert markdown bold (**text**) to <b>text</b> for HTML rendering
                    content = msg["content"].replace("\n", "<br>")
                    import re as _re
                    content = _re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', content)
                    st.markdown(ai_bubble(content), unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # ── Input row ─────────────────────────────────────────────────
            col_inp, col_btn = st.columns([5, 1])
            with col_inp:
                user_q = st.text_input(
                    "Ask a clinical question...",
                    key="chat_input",
                    label_visibility="collapsed",
                    placeholder="e.g. What medications should be considered? What does the ECG finding mean?")
            with col_btn:
                send = st.button("Send", use_container_width=True)

            if send and user_q.strip():
                engine = ClinicalEngine(pd_, pname, res, prob)
                answer = engine.answer_question(user_q)
                st.session_state["chat_history"].append({"role": "user",      "content": user_q})
                st.session_state["chat_history"].append({"role": "assistant", "content": answer})
                st.rerun()

        # ── Quick question buttons ─────────────────────────────────────────
        with ai_right:
            st.markdown('<div class="cardio-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Quick Questions</div>', unsafe_allow_html=True)

            quick_qs = [
                "What is the overall risk assessment?",
                "Explain the blood pressure reading",
                "What does the cholesterol level mean?",
                "Interpret the ST oldpeak & slope",
                "Explain the chest pain classification",
                "What do the ECG findings indicate?",
                "Significance of exercise angina?",
                "What do the vessel findings mean?",
                "Interpret the perfusion/thal result",
                "What medications are recommended?",
                "What lifestyle changes are needed?",
                "What is the long-term prognosis?",
                "Is a specialist referral needed?",
                "Show normal range comparisons",
                "Framingham 10-year CVD risk?",
            ]
            for q in quick_qs:
                if st.button(q, key=f"qq_{q[:20]}", use_container_width=True):
                    engine = ClinicalEngine(pd_, pname, res, prob)
                    answer = engine.answer_question(q)
                    st.session_state["chat_history"].append({"role": "user",      "content": q})
                    st.session_state["chat_history"].append({"role": "assistant", "content": answer})
                    st.rerun()

            if st.session_state["chat_history"]:
                st.markdown("---")
                if st.button("🗑 Clear Chat", use_container_width=True):
                    st.session_state["chat_history"] = []
                    st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — REPORT
# ═══════════════════════════════════════════════════════════════════════════
with tab_report:
    if not st.session_state["prediction_done"]:
        st.info("🫀 Run a prediction to generate a clinical report.")
    else:
        res   = st.session_state["result"]
        prob  = st.session_state["probability"]
        pd_   = st.session_state["patient_data"]
        pname = st.session_state["patient_name"]
        flags = st.session_state["risk_flags"]
        recs  = st.session_state["recommendations"]
        summ  = st.session_state["clinical_summary"]

        st.markdown('<div class="cardio-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Generate Clinical PDF Report</div>', unsafe_allow_html=True)
        st.markdown(
            f"Ready to export a comprehensive clinical report for **{pname}** "
            f"with all findings, risk flags, and recommendations.")

        if st.button("📄 Download PDF Report", use_container_width=False):
            pdf_bytes = generate_pdf_report(pname, pd_, res, prob, flags, recs, summ)
            b64   = base64.b64encode(pdf_bytes).decode()
            fname = f"HeartStrokePredictor_{pname.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf"
            href  = (f'<a href="data:application/pdf;base64,{b64}" download="{fname}" '
                     f'style="color:#00E5C3;font-weight:600;">'
                     f'⬇️ Click here if download does not start automatically</a>')
            st.markdown(href, unsafe_allow_html=True)
            st.success(f"✅ Report generated for {pname}")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="cardio-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Report Preview</div>', unsafe_allow_html=True)

        danger_flags = [(f, s) for f, s in flags if s == "danger"]
        warn_flags   = [(f, s) for f, s in flags if s == "warn"]

        col_a, col_b = st.columns(2)
        with col_a:
            tier_col = "#FF4D6D" if prob >= 65 else "#F59E0B" if prob >= 40 else "#10B981"
            st.markdown(
                f"**Patient:** {pname}  \n"
                f"**Age / Sex:** {pd_['age']} / {'Male' if pd_['sex'] else 'Female'}  \n"
                f"**Result:** <span style='color:{tier_col};font-weight:700;'>{res} ({prob}%)</span>  \n"
                f"**Date:** {datetime.now().strftime('%B %d, %Y')}",
                unsafe_allow_html=True)
        with col_b:
            st.markdown(f"**High-risk flags:** {len(danger_flags)}")
            st.markdown(f"**Moderate-risk flags:** {len(warn_flags)}")
            engine_p = ClinicalEngine(pd_, pname, res, prob)
            st.markdown(f"**Est. Framingham CVD Risk:** ~{engine_p._framingham_risk_estimate()}%")

        st.markdown("---")
        st.markdown("**Clinical Recommendations:**")
        for r in recs[:6]:
            st.markdown(r)
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center>"
    "🚀 Built with ❤️ by<br>"
    "<b>Ritesh Chaudhary</b> · "
    "<b>Gaurav Rai</b> · "
    "<b>Kumaresh Alu</b> · "
    "<b>Mohammad Zaid Ansari</b>"
    "</center>",
    unsafe_allow_html=True)