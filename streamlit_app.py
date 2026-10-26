"""
SCD-ActiPredict AI - Community SCD Older Adults Physical Inactivity Risk Prediction System
Production-Grade Streamlit Web Application
Compatible with Streamlit Community Cloud (GitHub: hhini/old_anactived)
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SCD-ActiPredict AI | 社区SCD老年人身体活动不足风险智能预测系统",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Theme Management
# ---------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "light"

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

IS_DARK = st.session_state.theme == "dark"

# ---------------------------------------------------------------------------
# Inject Advanced Big-Tech CSS Design System
# ---------------------------------------------------------------------------
CSS_LIGHT = """
<style>
:root {
    --bg-main: #f8fafc;
    --bg-card: #ffffff;
    --bg-subtle: #f1f5f9;
    --border-card: #e2e8f0;
    --border-hover: #cbd5e1;
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --text-muted: #64748b;
    --accent-primary: #4f46e5;
    --accent-light: #e0e7ff;
    --emerald: #10b981;
    --emerald-bg: rgba(16, 185, 129, 0.12);
    --amber: #f59e0b;
    --amber-bg: rgba(245, 158, 11, 0.12);
    --rose: #ef4444;
    --rose-bg: rgba(239, 68, 68, 0.12);
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.04);
}
</style>
"""

CSS_DARK = """
<style>
:root {
    --bg-main: #09090b;
    --bg-card: #121217;
    --bg-subtle: #18181f;
    --border-card: #27272f;
    --border-hover: #3f3f4e;
    --text-primary: #f8fafc;
    --text-secondary: #cbd5e1;
    --text-muted: #94a3b8;
    --accent-primary: #6366f1;
    --accent-light: rgba(99, 102, 241, 0.2);
    --emerald: #34d399;
    --emerald-bg: rgba(52, 211, 153, 0.15);
    --amber: #fbbf24;
    --amber-bg: rgba(251, 191, 36, 0.15);
    --rose: #f87171;
    --rose-bg: rgba(248, 113, 113, 0.15);
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.4);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.5), 0 2px 4px -2px rgba(0, 0, 0, 0.4);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.6), 0 4px 6px -4px rgba(0, 0, 0, 0.5);
}
</style>
"""

COMMON_CSS = """
<style>
/* Global App Canvas */
header[data-testid="stHeader"], footer, [data-testid="stToolbar"] {
    display: none !important;
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container {
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", "PingFang SC", "Microsoft YaHei", sans-serif !important;
    -webkit-font-smoothing: antialiased;
}
.block-container {
    padding: 1.5rem 2.5rem 3rem !important;
    max-width: 1400px !important;
}

/* Big-Tech Top Navbar */
.top-navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: 16px;
    padding: 1.1rem 1.6rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-md);
}
.brand-group {
    display: flex;
    align-items: center;
    gap: 14px;
}
.brand-icon {
    width: 44px;
    height: 44px;
    background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 22px;
    box-shadow: 0 4px 10px rgba(79, 70, 229, 0.3);
}
.brand-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.brand-subtitle {
    font-size: 0.8rem;
    color: var(--text-muted);
    font-weight: 500;
    margin-top: 2px;
}

/* Badges */
.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-champion {
    background: rgba(79, 70, 229, 0.12);
    color: var(--accent-primary);
    border: 1px solid rgba(79, 70, 229, 0.25);
}
.badge-low {
    background: var(--emerald-bg);
    color: var(--emerald);
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.badge-mod {
    background: var(--amber-bg);
    color: var(--amber);
    border: 1px solid rgba(245, 158, 11, 0.3);
}
.badge-high {
    background: var(--rose-bg);
    color: var(--rose);
    border: 1px solid rgba(239, 68, 68, 0.3);
}

/* Apple-style Cards */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: var(--shadow-sm);
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    margin-bottom: 1.25rem;
}
.glass-card:hover {
    border-color: var(--border-hover);
    box-shadow: var(--shadow-md);
}
.card-header-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.35rem;
}
.card-header-desc {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-bottom: 1.25rem;
}

/* Metric Display Blocks */
.hero-risk-box {
    text-align: center;
    padding: 1.6rem 1rem;
    border-radius: 16px;
    margin-bottom: 1.25rem;
    border: 1px solid var(--border-card);
}
.hero-risk-pct {
    font-size: 3.2rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    line-height: 1;
    margin: 0.5rem 0;
}
.hero-risk-label {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Section Header Typography */
.section-headline {
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: var(--text-primary);
    margin-bottom: 0.3rem;
}
.section-subtext {
    font-size: 0.88rem;
    color: var(--text-secondary);
    margin-bottom: 1.5rem;
}

/* Pill Tabs */
button[data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    padding: 0.65rem 1.4rem !important;
    border: 1px solid transparent !important;
    border-radius: 10px !important;
    transition: all 0.2s ease;
}
button[data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
    background: var(--bg-subtle) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #ffffff !important;
    background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35) !important;
}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {
    display: none !important;
}
[data-baseweb="tab-list"] {
    gap: 8px !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border-card) !important;
    border-radius: 14px !important;
    padding: 6px !important;
    margin-bottom: 1.8rem !important;
    box-shadow: var(--shadow-sm) !important;
}

/* Action button */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
</style>
"""

st.markdown(CSS_DARK if IS_DARK else CSS_LIGHT, unsafe_allow_html=True)
st.markdown(COMMON_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data & Model Loader (Cached & Resilient)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_prediction_resources():
    model_path = os.path.join("models", "best_model.pkl")
    scaler_path = os.path.join("models", "standard_scaler.pkl")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            return model, scaler, True
        except Exception as e:
            st.warning(f"Model load notice: {e}. Fallback formula enabled.")
            
    # Fallback to multivariable logistic regression parameters if file missing or version conflict
    return None, None, False

model, scaler, model_loaded = load_prediction_resources()

# Expected feature list in strict scaler order
FEATURE_COLS = [
    'Age', 'Monthly_Income', 'Trauma_History', 'BMI', 'Hemoglobin',
    'GSES_Score', 'Social_Activity_Freq', 'Sedentary_Hobby_Non_Intellectual',
    'SSPA_Score', 'Community_Facility_Score', 'Community_Activity_Score'
]

# Scaler fallback parameters (derived from 385 cohort)
SCALER_MEANS = {
    'Age': 70.286, 'Monthly_Income': 1.918, 'Trauma_History': 0.175, 'BMI': 22.180,
    'Hemoglobin': 127.004, 'GSES_Score': 18.937, 'Social_Activity_Freq': 0.978,
    'Sedentary_Hobby_Non_Intellectual': 0.257, 'SSPA_Score': 10.937,
    'Community_Facility_Score': 8.446, 'Community_Activity_Score': 4.717
}
SCALER_STDS = {
    'Age': 5.076, 'Monthly_Income': 1.060, 'Trauma_History': 0.380, 'BMI': 1.940,
    'Hemoglobin': 9.910, 'GSES_Score': 3.106, 'Social_Activity_Freq': 0.684,
    'Sedentary_Hobby_Non_Intellectual': 0.437, 'SSPA_Score': 3.829,
    'Community_Facility_Score': 1.621, 'Community_Activity_Score': 2.275
}

# Multivariable Logistic Regression Parametric Fallback (Table 5 MLE Parameters)
LOGIT_COEFS = {
    'const': -0.7986,
    'Sedentary_Hobby_Non_Intellectual': 1.7441,
    'Trauma_History': 1.6394,
    'BMI': 0.2200,
    'Hemoglobin': -0.0427,
    'Monthly_Income': -0.4350,
    'Age': 0.0603,
    'Community_Activity_Score': -0.1421,
    'Social_Activity_Freq': -0.3825,
    'Community_Facility_Score': -0.1600,
    'SSPA_Score': -0.0479,
    'GSES_Score': -0.0349
}

def predict_risk_probability(feature_dict):
    """Predicts physical inactivity risk probability using champion XGBoost or robust Logit fallback."""
    df_raw = pd.DataFrame([feature_dict])[FEATURE_COLS]
    
    if model_loaded:
        try:
            X_scaled = scaler.transform(df_raw)
            prob = float(model.predict_proba(X_scaled)[0, 1])
            return prob, "XGBoost (Champion ML)"
        except Exception:
            pass
            
    # Mathematical Logit Sigmoid Fallback
    lp = LOGIT_COEFS['const']
    for feat in FEATURE_COLS:
        lp += LOGIT_COEFS[feat] * df_raw.loc[0, feat]
    prob = 1.0 / (1.0 + np.exp(-lp))
    return float(prob), "Multivariable Logistic Regression (MLE)"

def compute_waterfall_contributions(feature_dict):
    """Calculates directional risk impact of each predictor against cohort baseline."""
    contributions = []
    
    # Biological / Clinical Directional Weights
    weights = {
        'Sedentary_Hobby_Non_Intellectual': ('静态非益智类爱好', 1.7441, SCALER_MEANS['Sedentary_Hobby_Non_Intellectual']),
        'Trauma_History': ('既往跌倒外伤史', 1.6394, SCALER_MEANS['Trauma_History']),
        'BMI': ('体质指数 (BMI)', 0.2200, SCALER_MEANS['BMI']),
        'Age': ('年龄（岁）', 0.0603, SCALER_MEANS['Age']),
        'Monthly_Income': ('人均月收入', -0.4350, SCALER_MEANS['Monthly_Income']),
        'Hemoglobin': ('血红蛋白水平', -0.0427, SCALER_MEANS['Hemoglobin']),
        'Community_Facility_Score': ('社区设施支持', -0.1600, SCALER_MEANS['Community_Facility_Score']),
        'Community_Activity_Score': ('社区活动支持', -0.1421, SCALER_MEANS['Community_Activity_Score']),
        'Social_Activity_Freq': ('社交活动频率', -0.3825, SCALER_MEANS['Social_Activity_Freq']),
        'SSPA_Score': ('社会支持总分', -0.0479, SCALER_MEANS['SSPA_Score']),
        'GSES_Score': ('自我效能感', -0.0349, SCALER_MEANS['GSES_Score']),
    }
    
    for k, (name_cn, coef, base_val) in weights.items():
        delta = (feature_dict[k] - base_val) * coef
        contributions.append({
            'Feature': name_cn,
            'Delta': delta,
            'Is_Risk': delta > 0,
            'Magnitude': abs(delta)
        })
        
    df_contr = pd.DataFrame(contributions).sort_values(by='Magnitude', ascending=True)
    return df_contr

# ---------------------------------------------------------------------------
# Top Floating Navigation Bar
# ---------------------------------------------------------------------------
nav_c1, nav_c2 = st.columns([10, 2])
with nav_c1:
    st.markdown(f"""
    <div class="top-navbar">
        <div class="brand-group">
            <div class="brand-icon">🏃‍♂️</div>
            <div>
                <div class="brand-title">SCD-ActiPredict AI · 社区SCD老年人身体活动不足风险智能预测系统</div>
                <div class="brand-subtitle">国家老年疾病临床医学研究网络规范 · TRIPOD+AI 2024 · 赣州前瞻队列 (N=385)</div>
            </div>
        </div>
        <div style="display:flex; gap:10px; align-items:center;">
            <span class="badge-pill badge-champion">🟢 XGBoost 冠军模型 (AUC 0.832)</span>
            <span class="badge-pill badge-pill" style="background:var(--bg-subtle); color:var(--text-secondary); border:1px solid var(--border-card);">🎯 灵敏度 75.6%</span>
            <a href="https://github.com/hhini/old_anactived" target="_blank" style="text-decoration:none;">
                <span class="badge-pill badge-pill" style="background:rgba(79,70,229,0.1); color:var(--accent-primary); border:1px solid rgba(79,70,229,0.3);">💻 GitHub 源码</span>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

with nav_c2:
    theme_text = "☀️ 浅色模式" if IS_DARK else "🌙 深色模式"
    st.button(theme_text, on_click=toggle_theme, use_container_width=True)

# ---------------------------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------------------------
tab_calc, tab_arena, tab_xai, tab_guideline = st.tabs([
    "🧮 临床智能风险计算器 (Risk Calculator)",
    "🏆 七大算法竞技场 (Model Arena)",
    "🔬 特征归因与列线图 (XAI & Nomogram)",
    "📘 社区分级防控与干预指南 (Action Plan)"
])

# ===========================================================================
# TAB 1: CLINICAL RISK CALCULATOR
# ===========================================================================
with tab_calc:
    st.markdown('<div class="section-headline">社区老年主观认知下降（SCD）个体化身体活动不足风险智能测算</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">输入受试老年人 11 项核心生物医学与社会生态学指标，系统实时基于最佳 XGBoost 算法测算活动不足风险概率，触发双轨决策切点预警，并开具反事实生活方式处方。</div>', unsafe_allow_html=True)
    
    # Case Presets Bar
    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
    with preset_col1:
        if st.button("🔴 载入典型高危病例 (Case A)", use_container_width=True):
            st.session_state.inp_age = 74
            st.session_state.inp_inc = 0 # <1000元
            st.session_state.inp_trauma = 1 # 有外伤
            st.session_state.inp_bmi = 28.5
            st.session_state.inp_hb = 112.0 # 贫血
            st.session_state.inp_gses = 14.0 # 低效能
            st.session_state.inp_social = 0 # 偶尔
            st.session_state.inp_hobby = 1 # 有静态爱好
            st.session_state.inp_sspa = 6.0 # 低支持
            st.session_state.inp_fac = 5.0 # 设施一般
            st.session_state.inp_act = 3.0 # 活动支持弱
    with preset_col2:
        if st.button("🟢 载入典型低危老人 (Case B)", use_container_width=True):
            st.session_state.inp_age = 65
            st.session_state.inp_inc = 3 # 3000~4999元
            st.session_state.inp_trauma = 0 # 无外伤
            st.session_state.inp_bmi = 21.5
            st.session_state.inp_hb = 138.0 # 正常
            st.session_state.inp_gses = 26.0 # 高效能
            st.session_state.inp_social = 2 # 经常
            st.session_state.inp_hobby = 0 # 无静态爱好
            st.session_state.inp_sspa = 18.0 # 良好支持
            st.session_state.inp_fac = 11.0 # 优质设施
            st.session_state.inp_act = 10.0 # 丰富组织
    with preset_col3:
        if st.button("🟡 载入中危亚健康过渡老人", use_container_width=True):
            st.session_state.inp_age = 70
            st.session_state.inp_inc = 2 # 1000~2999元
            st.session_state.inp_trauma = 0
            st.session_state.inp_bmi = 24.2
            st.session_state.inp_hb = 126.0
            st.session_state.inp_gses = 19.0
            st.session_state.inp_social = 1 # 有时
            st.session_state.inp_hobby = 1
            st.session_state.inp_sspa = 11.0
            st.session_state.inp_fac = 8.0
            st.session_state.inp_act = 5.0
    with preset_col4:
        if st.button("🔄 重置为社区中位基线值", use_container_width=True):
            st.session_state.inp_age = 71
            st.session_state.inp_inc = 2
            st.session_state.inp_trauma = 0
            st.session_state.inp_bmi = 22.2
            st.session_state.inp_hb = 127.0
            st.session_state.inp_gses = 19.0
            st.session_state.inp_social = 1
            st.session_state.inp_hobby = 0
            st.session_state.inp_sspa = 11.0
            st.session_state.inp_fac = 8.0
            st.session_state.inp_act = 5.0

    # Initialize session state if not set
    if "inp_age" not in st.session_state:
        st.session_state.inp_age = 72
        st.session_state.inp_inc = 2
        st.session_state.inp_trauma = 1
        st.session_state.inp_bmi = 24.8
        st.session_state.inp_hb = 122.0
        st.session_state.inp_gses = 18.0
        st.session_state.inp_social = 1
        st.session_state.inp_hobby = 1
        st.session_state.inp_sspa = 10.0
        st.session_state.inp_fac = 8.0
        st.session_state.inp_act = 4.0

    calc_left, calc_right = st.columns([7, 5])
    
    with calc_left:
        # Card 1: Demographics & Physiological Health
        st.markdown("""
        <div class="glass-card">
            <div class="card-header-title">🩺 生理机能与健康状况 (Physiological & Clinical Factors)</div>
            <div class="card-header-desc">涵盖心肺与代谢负荷、机体耐力底座及关节骨骼损伤病史</div>
        """, unsafe_allow_html=True)
        
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            inp_age = st.slider("1. 年龄（岁）", min_value=60, max_value=90, value=int(st.session_state.inp_age), step=1, help="增龄导致骨骼肌纤维萎缩与心肺摄氧量自然衰退 (aOR=1.06)")
            inp_bmi = st.slider("2. 体质指数 BMI (kg/m²)", min_value=16.0, max_value=35.0, value=float(st.session_state.inp_bmi), step=0.1, help="BMI每增加1 kg/m²，关节机械负荷与活动不足几率递增24.6% (aOR=1.25)")
            inp_hb = st.slider("3. 血红蛋白水平 (g/L)", min_value=90.0, max_value=170.0, value=float(st.session_state.inp_hb), step=1.0, help="机体组织携氧与有氧运动耐力底座，<120 g/L提示贫血耐力受损 (aOR=0.96)")
        with c_p2:
            trauma_opts = {0: "无外伤跌倒史 (No)", 1: "有跌倒外伤或骨折史 (Yes)"}
            inp_trauma = st.radio("4. 既往外伤骨折病史", options=[0, 1], format_func=lambda x: trauma_opts[x], index=int(st.session_state.inp_trauma), help="跌倒外伤遗留慢性关节疼痛并激发强烈的'跌倒恐惧综合征' (aOR=5.15)")
            inc_opts = {0: "<1000 元/月", 1: "1000~1999 元/月", 2: "2000~2999 元/月", 3: "3000~4999 元/月", 4: "≥5000 元/月"}
            inp_inc = st.selectbox("5. 家庭人均月收入阶梯", options=[0, 1, 2, 3, 4], format_func=lambda x: inc_opts[x], index=int(st.session_state.inp_inc), help="经济资本充裕度赋予更强的健康投入与文体健身参与能力 (aOR=0.65)")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Card 2: Behavioral & Social Ecological Factors
        st.markdown("""
        <div class="glass-card">
            <div class="card-header-title">🏘️ 行为生活方式、心理与社区环境 (Behavioral & Ecological Factors)</div>
            <div class="card-header-desc">涵盖日常休闲习惯、社会支持网络、自我效能信念与社区适老设施环境</div>
        """, unsafe_allow_html=True)
        
        c_b1, c_b2 = st.columns(2)
        with c_b1:
            hobby_opts = {0: "否 (无静态不良爱好，平时喜爱散步/太极/阅读等)", 1: "是 (长期久坐看电视、闲坐发呆等静态非益智类爱好)"}
            inp_hobby = st.radio("6. 静态非益智类休闲爱好", options=[0, 1], format_func=lambda x: hobby_opts[x], index=int(st.session_state.inp_hobby), help="全模型最强独立致病危险因素！低脑耗娱乐挤占运动物理时间窗口并钝化多巴胺犒赏 (aOR=5.72)")
            soc_opts = {0: "偶尔参与 (Seldom)", 1: "有时参与 (Sometimes)", 2: "经常参与 (Often)"}
            inp_social = st.selectbox("7. 社交活动参与频率", options=[0, 1, 2], format_func=lambda x: soc_opts[x], index=int(st.session_state.inp_social), help="融入社区熟人社群能显著提高结伴出行与集体跳操几率")
            inp_gses = st.slider("8. GSES 一般自我效能感得分 (10~30分)", min_value=10.0, max_value=30.0, value=float(st.session_state.inp_gses), step=0.5, help="个体克服关节酸痛、天气不佳等阻力坚持运动的内在心理掌控信念")
        with c_b2:
            inp_sspa = st.slider("9. SSPA 体力活动社会支持得分 (5~25分)", min_value=5.0, max_value=25.0, value=float(st.session_state.inp_sspa), step=0.5, help="来自配偶、子女及同伴的情感鼓励、生活照料与共同锻炼督导 (≥10分为良好)")
            inp_fac = st.slider("10. 社区适老健身设施支持得分 (3~12分)", min_value=3.0, max_value=12.0, value=float(st.session_state.inp_fac), step=0.5, help="社区步道平整度、无障碍设施、健身器械丰富度及遮阳避雨休息点可及性")
            inp_act = st.slider("11. 社区活动组织支持得分 (3~12分)", min_value=3.0, max_value=12.0, value=float(st.session_state.inp_act), step=0.5, help="居委会与社区卫生服务中心定期组织的老年体艺比赛、八段锦太极培训频度")
        st.markdown("</div>", unsafe_allow_html=True)

    # Compile current inputs
    current_patient = {
        'Age': float(inp_age),
        'Monthly_Income': float(inp_inc),
        'Trauma_History': float(inp_trauma),
        'BMI': float(inp_bmi),
        'Hemoglobin': float(inp_hb),
        'GSES_Score': float(inp_gses),
        'Social_Activity_Freq': float(inp_social),
        'Sedentary_Hobby_Non_Intellectual': float(inp_hobby),
        'SSPA_Score': float(inp_sspa),
        'Community_Facility_Score': float(inp_fac),
        'Community_Activity_Score': float(inp_act)
    }

    # Execute Prediction
    risk_prob, model_used_name = predict_risk_probability(current_patient)
    risk_pct = risk_prob * 100.0

    # Determine Clinical Risk Stratification
    if risk_prob < 0.20:
        tier_label = "🟢 低危梯队 (Low Risk Tier)"
        tier_color = "var(--emerald)"
        tier_bg = "var(--emerald-bg)"
        badge_cls = "badge-low"
        advice_text = "该老年人机能储备充沛、环境支持完善，预测活动不足发生率极低。建议维持现有生活形态，纳入社区常规年度健康监测即可。"
    elif risk_prob < 0.50:
        tier_label = "🟡 中危梯队 (Moderate Risk Tier)"
        tier_color = "var(--amber)"
        tier_bg = "var(--amber-bg)"
        badge_cls = "badge-mod"
        advice_text = "属于亚健康行为滑坡过渡群！已突破社区普查切点 (0.270)。建议家庭医生及时签约介入，重点防范久坐固化，推行'坐45动5分'微运动方案。"
    else:
        tier_label = "🔴 高危梯队 (High Risk Tier)"
        tier_color = "var(--rose)"
        tier_bg = "var(--rose-bg)"
        badge_cls = "badge-high"
        advice_text = "已突破临床确诊切点 (0.500)！存在严重久坐废用与跌倒恐惧风险，必须立即启动专科与家庭医生强化干预，落实反事实个性化处方！"

    with calc_right:
        # Prediction Output Hero Card
        st.markdown(f"""
        <div class="glass-card" style="border-top: 4px solid {tier_color};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="card-header-title">📊 预测结果看板</span>
                <span class="badge-pill {badge_cls}">{tier_label}</span>
            </div>
            <div class="hero-risk-box" style="background:{tier_bg};">
                <div class="hero-risk-label" style="color:{tier_color};">个体身体活动不足预测概率</div>
                <div class="hero-risk-pct" style="color:{tier_color};">{risk_pct:.1f}%</div>
                <div style="font-size:0.75rem; color:var(--text-muted);">计算引擎: {model_used_name}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Dual-Threshold Indicator
        th_c1, th_c2 = st.columns(2)
        with th_c1:
            is_screen_alert = risk_prob >= 0.270
            sc_color = "var(--amber)" if is_screen_alert else "var(--emerald)"
            sc_text = "⚠️ 超出切点 (预警)" if is_screen_alert else "✅ 正常安全区"
            st.markdown(f"""
            <div style="border:1px solid var(--border-card); border-radius:10px; padding:8px 12px; background:var(--bg-subtle);">
                <div style="font-size:0.72rem; color:var(--text-muted); font-weight:600;">社区普查切点 (0.270)</div>
                <div style="font-size:0.85rem; font-weight:700; color:{sc_color};">{sc_text}</div>
            </div>
            """, unsafe_allow_html=True)
        with th_c2:
            is_clinic_alert = risk_prob >= 0.500
            cl_color = "var(--rose)" if is_clinic_alert else "var(--emerald)"
            cl_text = "🚨 超出切点 (强化)" if is_clinic_alert else "✅ 正常安全区"
            st.markdown(f"""
            <div style="border:1px solid var(--border-card); border-radius:10px; padding:8px 12px; background:var(--bg-subtle);">
                <div style="font-size:0.72rem; color:var(--text-muted); font-weight:600;">临床确诊切点 (0.500)</div>
                <div style="font-size:0.85rem; font-weight:700; color:{cl_color};">{cl_text}</div>
            </div>
            """, unsafe_allow_html=True)

        # Plotly Semi-Circle Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            number={'suffix': "%", 'font': {'size': 28, 'color': tier_color, 'family': "DM Sans"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#71717a", 'tickmode': "array", 'tickvals': [0, 20, 27, 50, 75, 100]},
                'bar': {'color': tier_color, 'thickness': 0.28},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 1,
                'bordercolor': "#cbd5e1" if not IS_DARK else "#334155",
                'steps': [
                    {'range': [0, 20], 'color': 'rgba(16, 185, 129, 0.22)'},
                    {'range': [20, 50], 'color': 'rgba(245, 158, 11, 0.22)'},
                    {'range': [50, 100], 'color': 'rgba(239, 68, 68, 0.22)'}
                ],
                'threshold': {
                    'line': {'color': "#dc2626", 'width': 3},
                    'thickness': 0.85,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=210,
            margin=dict(l=20, r=20, t=10, b=0)
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
        
        st.markdown(f"""
        <div style="font-size:0.82rem; color:var(--text-secondary); line-height:1.5; padding:8px 12px; background:var(--bg-subtle); border-radius:10px; margin-top:5px;">
            <b>临床决策指南：</b>{advice_text}
        </div>
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Interactive Waterfall Breakdown & Counterfactual Prescription Simulator
    # -----------------------------------------------------------------------
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    c_dec1, c_dec2 = st.columns([6, 6])
    
    with c_dec1:
        st.markdown("""
        <div class="glass-card">
            <div class="card-header-title">🔍 单病例驱动特征归因分解 (Patient-Specific Contribution)</div>
            <div class="card-header-desc">展示当前老人各指标偏离基线均值对发病对数几率的增减边际影响（红色推高风险，绿色降低风险）</div>
        """, unsafe_allow_html=True)
        
        df_contr = compute_waterfall_contributions(current_patient)
        
        # Color bar: Red for risk increase, Emerald for protection
        bar_colors = ['#ef4444' if r else '#10b981' for r in df_contr['Is_Risk']]
        
        fig_contr = go.Figure(go.Bar(
            y=df_contr['Feature'],
            x=df_contr['Delta'],
            orientation='h',
            marker=dict(color=bar_colors),
            text=[f"{v:+.2f}" for v in df_contr['Delta']],
            textposition='auto',
            textfont=dict(size=10, color="white", family="DM Sans")
        ))
        fig_contr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#64748b" if not IS_DARK else "#94a3b8", size=11),
            height=340,
            margin=dict(l=10, r=10, t=10, b=25),
            xaxis=dict(
                title="边际风险贡献倾向 (Marginal Log-Odds Impact)",
                zeroline=True,
                zerolinecolor="#94a3b8",
                gridcolor="rgba(0,0,0,0.05)" if not IS_DARK else "rgba(255,255,255,0.05)"
            ),
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_contr, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c_dec2:
        st.markdown("""
        <div class="glass-card">
            <div class="card-header-title">⚡ 反事实生活方式处方模拟器 (Counterfactual Optimizer)</div>
            <div class="card-header-desc">保持不可逆生物学特征不变，模拟调控生活方式与社区环境后预测发病风险的逆转路径</div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='font-size:0.83rem; font-weight:600; margin-bottom:8px;'>请勾选拟实施的精准干预方案：</div>", unsafe_allow_html=True)
        sim_hobby = st.checkbox("✅ 破除久坐链条：彻底戒除看电视闲坐等静态非益智爱好", value=(inp_hobby == 1))
        sim_fac = st.checkbox("✅ 社区环境赋能：提高适老化设施支持得分至 10.0 分（增设步道器械）", value=(inp_fac < 9.0))
        sim_sspa = st.checkbox("✅ 家庭社会动员：强化配偶与子女督导，社会支持提至 18.0 分", value=(inp_sspa < 14.0))
        sim_act = st.checkbox("✅ 组织社团活动：定期参与居委会太极/八段锦集体活动至 9.0 分", value=(inp_act < 8.0))
        
        # Build Simulated Patient
        sim_patient = current_patient.copy()
        if sim_hobby:
            sim_patient['Sedentary_Hobby_Non_Intellectual'] = 0.0
        if sim_fac:
            sim_patient['Community_Facility_Score'] = max(current_patient['Community_Facility_Score'], 10.0)
        if sim_sspa:
            sim_patient['SSPA_Score'] = max(current_patient['SSPA_Score'], 18.0)
        if sim_act:
            sim_patient['Community_Activity_Score'] = max(current_patient['Community_Activity_Score'], 9.0)
            sim_patient['Social_Activity_Freq'] = max(current_patient['Social_Activity_Freq'], 2.0)
            
        sim_risk, _ = predict_risk_probability(sim_patient)
        sim_risk_pct = sim_risk * 100.0
        risk_diff = sim_risk_pct - risk_pct
        
        # Comparison Visual
        fig_cf = go.Figure()
        fig_cf.add_trace(go.Bar(
            name='干预前初始风险',
            x=['身体活动不足发病几率'],
            y=[risk_pct],
            marker_color='#ef4444',
            text=[f"{risk_pct:.1f}%"],
            textposition='auto',
            width=0.35
        ))
        fig_cf.add_trace(go.Bar(
            name='反事实处方干预后',
            x=['身体活动不足发病几率'],
            y=[sim_risk_pct],
            marker_color='#10b981',
            text=[f"{sim_risk_pct:.1f}%"],
            textposition='auto',
            width=0.35
        ))
        fig_cf.update_layout(
            barmode='group',
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#64748b" if not IS_DARK else "#94a3b8", size=11),
            height=180,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(range=[0, 100], ticksuffix="%"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cf, use_container_width=True, config={"displayModeBar": False})
        
        if risk_diff < 0:
            st.success(f"🎉 处方优化成功！预计发病风险从 **{risk_pct:.1f}%** 显著骤降至 **{sim_risk_pct:.1f}%** (绝对降幅: **{abs(risk_diff):.1f}%**)，成功将老人拉回临床安全保护区！")
        else:
            st.info("提示：请在上方勾选可行的干预动作以测算风险压降效益。")
            
        st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================================
# TAB 2: MULTI-MODEL ARENA & BENCHMARK
# ===========================================================================
with tab_arena:
    st.markdown('<div class="section-headline">七大主流机器学习算法建模竞技与独立测试集盲态评测</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">严格遵循 TRIPOD+AI 2024 规范，在独立验证测试集（n=116，活动不足 41 例，充足 75 例）上对 7 大算法执行 ROC、PR、校准度、F1 分数及临床净获益全维度客观量化横评。</div>', unsafe_allow_html=True)
    
    # KPI Highlights
    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
    with m_c1:
        st.markdown("""
        <div class="glass-card" style="border-left:4px solid #4f46e5;">
            <div style="font-size:0.75rem; color:var(--text-muted); font-weight:600;">🏆 冠军模型 (Champion)</div>
            <div style="font-size:1.6rem; font-weight:800; color:var(--text-primary); margin:4px 0;">XGBoost</div>
            <div class="badge-pill badge-champion">测试 AUC: 0.8319</div>
        </div>
        """, unsafe_allow_html=True)
    with m_c2:
        st.markdown("""
        <div class="glass-card" style="border-left:4px solid #10b981;">
            <div style="font-size:0.75rem; color:var(--text-muted); font-weight:600;">🎯 最高召回/敏感度 (Recall)</div>
            <div style="font-size:1.6rem; font-weight:800; color:var(--text-primary); margin:4px 0;">75.61%</div>
            <div class="badge-pill badge-low">高危检出率全面占优</div>
        </div>
        """, unsafe_allow_html=True)
    with m_c3:
        st.markdown("""
        <div class="glass-card" style="border-left:4px solid #06b6d4;">
            <div style="font-size:0.75rem; color:var(--text-muted); font-weight:600;">⚖️ 综合调和 F1 分数</div>
            <div style="font-size:1.6rem; font-weight:800; color:var(--text-primary); margin:4px 0;">0.7126</div>
            <div class="badge-pill" style="background:rgba(6,182,212,0.12); color:#06b6d4;">精确率与召回率平衡</div>
        </div>
        """, unsafe_allow_html=True)
    with m_c4:
        st.markdown("""
        <div class="glass-card" style="border-left:4px solid #f59e0b;">
            <div style="font-size:0.75rem; color:var(--text-muted); font-weight:600;">📈 信号饱和收敛带</div>
            <div style="font-size:1.6rem; font-weight:800; color:var(--text-primary); margin:4px 0;">0.803 ~ 0.832</div>
            <div class="badge-pill badge-mod">极差仅 0.029 (信号充分释放)</div>
        </div>
        """, unsafe_allow_html=True)

    # Multi-Model Leaderboard Table
    df_models_path = os.path.join("results", "08_multimodel_performance_comparison.csv")
    if os.path.exists(df_models_path):
        df_models = pd.read_csv(df_models_path)
        st.markdown("""
        <div class="glass-card">
            <div class="card-header-title">📋 7大经典算法独立验证集全维度量化指标排行榜</div>
        """, unsafe_allow_html=True)
        st.dataframe(
            df_models[['Model', 'Test_ROC_AUC', 'Test_AUC_95CI', 'Sensitivity_Recall', 'Specificity', 'PPV_Precision', 'F1_Score', 'Brier_Score']],
            use_container_width=True,
            column_config={
                "Model": "算法名称",
                "Test_ROC_AUC": st.column_config.NumberColumn("ROC-AUC", format="%.4f"),
                "Test_AUC_95CI": "95% 置信区间",
                "Sensitivity_Recall": st.column_config.NumberColumn("敏感度 (Sensitivity)", format="%.2f%%"),
                "Specificity": st.column_config.NumberColumn("特异度 (Specificity)", format="%.2f%%"),
                "PPV_Precision": st.column_config.NumberColumn("精确率 (Precision)", format="%.2f%%"),
                "F1_Score": st.column_config.NumberColumn("F1 分数", format="%.4f"),
                "Brier_Score": st.column_config.NumberColumn("Brier 校准分数", format="%.4f"),
            },
            hide_index=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Model Performance Interactive Visualizations
    ar_c1, ar_c2 = st.columns(2)
    
    with ar_c1:
        st.markdown("""
        <div class="glass-card">
            <div class="card-header-title">📈 7大主流算法独立测试集 ROC 曲线全景对比</div>
            <div class="card-header-desc">XGBoost (AUC=0.832) 与 Logistic回归 (AUC=0.819) 全程领跑第一梯队</div>
        """, unsafe_allow_html=True)
        fig_roc_img = os.path.join("figures", "Fig4_MultiModel_ROC_Curves_Train_and_Test.png")
        if os.path.exists(fig_roc_img):
            st.image(fig_roc_img, use_container_width=True)
        else:
            st.info("ROC 曲线图生成中...")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with ar_c2:
        st.markdown("""
        <div class="glass-card">
            <div class="card-header-title">🕸️ 六维性能雷达竞技场 (Multi-Model Radar Comparison)</div>
            <div class="card-header-desc">辨别力、精确度、召回率、F1、特异度与校准度全景雷达覆盖面</div>
        """, unsafe_allow_html=True)
        fig_radar_img = os.path.join("figures", "Fig16_MultiModel_Radar_Performance_Comparison.png")
        if os.path.exists(fig_radar_img):
            st.image(fig_radar_img, use_container_width=True)
        else:
            st.info("雷达图生成中...")
        st.markdown("</div>", unsafe_allow_html=True)

    # Second row of arena
    ar_c3, ar_c4 = st.columns(2)
    with ar_c3:
        st.markdown("""
        <div class="glass-card">
            <div class="card-header-title">🎯 7大模型混淆矩阵大拼图 (Confusion Matrices Mosaic)</div>
            <div class="card-header-desc">深入剖析各算法在真阳性 (TP)、假阳性 (FP) 与假阴性 (FN) 上的临床权衡</div>
        """, unsafe_allow_html=True)
        fig_cm_img = os.path.join("figures", "Fig15_MultiModel_Confusion_Matrices_Mosaic.png")
        if os.path.exists(fig_cm_img):
            st.image(fig_cm_img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with ar_c4:
        st.markdown("""
        <div class="glass-card">
            <div class="card-header-title">🪜 三阶临床风险分层与事件单调校准 (Risk Stratification)</div>
            <div class="card-header-desc">证实低危 (<20%)、中危 (20%~50%) 与高危 (≥50%) 在实际发病率上呈现严格单调递增</div>
        """, unsafe_allow_html=True)
        fig_strat_img = os.path.join("figures", "Fig17_Clinical_Risk_Stratification_and_Outcomes.png")
        if os.path.exists(fig_strat_img):
            st.image(fig_strat_img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================================
# TAB 3: XAI INTERPRETABILITY & NOMOGRAM
# ===========================================================================
with tab_xai:
    st.markdown('<div class="section-headline">模型全局归因与 6 项核心独立特征深度透视 (SHAP & Nomogram)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">打破机器学习黑箱。结合多因素 Logistic 回归参数估计与博弈论最优解 SHAP 归因，阐明驱动社区SCD老年人活动行为的核心推拉动力学机制。</div>', unsafe_allow_html=True)
    
    # Top 6 Screened Features Cards
    st.markdown("<div style='font-size:1.05rem; font-weight:700; margin-bottom:12px;'>🌟 筛选确立的 6 项核心独立影响特征（4大危险因素 + 2大保护因素）</div>", unsafe_allow_html=True)
    
    f_c1, f_c2 = st.columns(2)
    with f_c1:
        st.markdown("""
        <div class="glass-card" style="border-left: 4px solid #ef4444;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-weight:700; color:#ef4444; font-size:1rem;">🔴 独立危险因素 1: 静态非益智类爱好</span>
                <span class="badge-pill badge-high">aOR = 5.72 (P < 0.001)</span>
            </div>
            <div style="font-size:0.83rem; color:var(--text-secondary); margin-top:6px; line-height:1.5;">
                <b>偏回归系数 β = +1.7441</b>。长期沉溺于久坐看电视、闲坐发呆等非益智爱好的SCD老年人，活动不足风险激增 4.72 倍。低脑耗娱乐挤占中高强度运动时间窗口并导致多巴胺犒赏系统钝化。
            </div>
        </div>
        <div class="glass-card" style="border-left: 4px solid #ef4444;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-weight:700; color:#ef4444; font-size:1rem;">🔴 独立危险因素 2: 既往外伤骨折史</span>
                <span class="badge-pill badge-high">aOR = 5.15 (P < 0.001)</span>
            </div>
            <div style="font-size:0.83rem; color:var(--text-secondary); margin-top:6px; line-height:1.5;">
                <b>偏回归系数 β = +1.6394</b>。外伤史不仅遗留关节慢性疼痛与本体感觉障碍，更诱发老年医学中著名的“跌倒恐惧综合征”（Fear of Falling），导致保护性行为严重退缩。
            </div>
        </div>
        <div class="glass-card" style="border-left: 4px solid #f59e0b;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-weight:700; color:#f59e0b; font-size:1rem;">🔴 独立危险因素 3: 高体质指数 (BMI)</span>
                <span class="badge-pill badge-mod">aOR = 1.25 (P = 0.002)</span>
            </div>
            <div style="font-size:0.83rem; color:var(--text-secondary); margin-top:6px; line-height:1.5;">
                <b>偏回归系数 β = +0.2200</b>。BMI 每增加 1 kg/m²，活动不足风险独立上升 24.6%。超重肥胖显著加重下肢关节机械应力并促进慢性神经炎症释放。
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with f_c2:
        st.markdown("""
        <div class="glass-card" style="border-left: 4px solid #f59e0b;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-weight:700; color:#f59e0b; font-size:1rem;">🔴 独立危险因素 4: 高龄累积效应</span>
                <span class="badge-pill badge-mod">aOR = 1.06 (P = 0.029)</span>
            </div>
            <div style="font-size:0.83rem; color:var(--text-secondary); margin-top:6px; line-height:1.5;">
                <b>偏回归系数 β = +0.0603</b>。呈现渐进性自然累积衰退效应，年龄每增加1岁，活动不足几率递增 6.2%；真实反映骨骼肌衰退与摄氧量自然下降规律。
            </div>
        </div>
        <div class="glass-card" style="border-left: 4px solid #10b981;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-weight:700; color:#10b981; font-size:1rem;">🟢 独立保护因素 1: 人均月收入良好</span>
                <span class="badge-pill badge-low">aOR = 0.65 (P = 0.004)</span>
            </div>
            <div style="font-size:0.83rem; color:var(--text-secondary); margin-top:6px; line-height:1.5;">
                <b>偏回归系数 β = -0.4350</b>。收入阶梯每上升一档，活动不足风险压降 35.3%。充沛的经济资本赋予更强的主动健康投资能力与文体参与条件。
            </div>
        </div>
        <div class="glass-card" style="border-left: 4px solid #10b981;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-weight:700; color:#10b981; font-size:1rem;">🟢 独立保护因素 2: 正常血红蛋白水平</span>
                <span class="badge-pill badge-low">aOR = 0.96 (P = 0.004)</span>
            </div>
            <div style="font-size:0.83rem; color:var(--text-secondary); margin-top:6px; line-height:1.5;">
                <b>偏回归系数 β = -0.0427</b>。外周血红蛋白每提升 1 g/L，活动不足风险下降 4.2%。血红蛋白是有氧代谢携氧与运动耐受力的生理底座。
            </div>
        </div>
        """, unsafe_allow_html=True)

    # SHAP Visualizations
    xai_c1, xai_c2 = st.columns(2)
    with xai_c1:
        st.markdown("""
        <div class="glass-card">
            <div class="card-header-title">📊 6大核心独立特征 SHAP 重要性柱状图与蜂群图 (Fig 8B)</div>
            <div class="card-header-desc">展示 6 项关键特征在树模型中的边际绝对重要度及特征值高低与风险方向的对应</div>
        """, unsafe_allow_html=True)
        fig_shap6_img = os.path.join("figures", "Fig8B_SHAP_Global_Summary_and_Beeswarm_Screened_6_Factors.png")
        if os.path.exists(fig_shap6_img):
            st.image(fig_shap6_img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with xai_c2:
        st.markdown("""
        <div class="glass-card">
            <div class="card-header-title">🌲 6大核心独立影响因素精选森林图 (Fig 3B)</div>
            <div class="card-header-desc">高精度展示 4 大危险因素与 2 大保护因素的调整后 aOR 与 95% 置信区间</div>
        """, unsafe_allow_html=True)
        fig_f6_img = os.path.join("figures", "Fig3B_Forest_Plot_Screened_6_Independent_Factors.png")
        if os.path.exists(fig_f6_img):
            st.image(fig_f6_img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Clinical Nomogram
    st.markdown("""
    <div class="glass-card">
        <div class="card-header-title">📋 临床床旁列线图 (Clinical Diagnostic Nomogram, Fig 11)</div>
        <div class="card-header-desc">严格基于多因素 Logistic 回归最终筛选确立的 6 项独立影响因素（静态非益智爱好、外伤史、BMI、年龄、血红蛋白、月收入），以 BMI 为 100 分基准标尺，单项得分垂直累加，总分轴 0~450 分与 Logistic 回归 Sigmoid 概率 100% 绝对闭环</div>
    """, unsafe_allow_html=True)
    fig_nomo_img = os.path.join("figures", "Fig11_Clinical_Diagnostic_Nomogram.png")
    if os.path.exists(fig_nomo_img):
        st.image(fig_nomo_img, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================================
# TAB 4: COMMUNITY ACTION PLAN & GUIDELINES
# ===========================================================================
with tab_guideline:
    st.markdown('<div class="section-headline">社区“防-筛-控”一体化行动图谱与适老化微运动干预指南</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">基于预测模型科研成果打通基层公卫转化“最后一公里”，构建面向社区卫生服务中心、家庭医生签约团队与社区网格员的落地操作规程。</div>', unsafe_allow_html=True)
    
    g_c1, g_c2 = st.columns(2)
    with g_c1:
        st.markdown("""
        <div class="glass-card">
            <div class="card-header-title">🏛️ 社区“防-筛-控”三级分层管理网络</div>
            <div style="font-size:0.85rem; color:var(--text-secondary); line-height:1.6; margin-top:8px;">
                <p><b>1. 一级预防（普遍宣教与环境支持）</b><br>
                面向全体社区SCD老人，强化适老无障碍步道建设，改善社区公共健身设施。倡导'坐四五、动五分'原则，打断连续久坐代谢停滞。</p>
                <p><b>2. 二级预防（早期普查与黄色预警）</b><br>
                利用本系统 <b>0.270 社区普查切点</b>，在老年人体检与慢病随访时初筛。超过切点者发放《防久坐健康宣教折页》，推荐参与社区'银发健康活力营'。</p>
                <p><b>3. 三级预防（专科介入与红色管控）</b><br>
                对评分 <b>≥ 0.500（高危梯队）</b> 老人触发红色警报。由全科医生与康复师联合介入，排查外伤后遗关节痛与贫血耐力问题，签署家庭医生定制生活方式处方。</p>
            </div>
        </div>
        <div class="glass-card">
            <div class="card-header-title">📋 基层医护人员三步标准化操作指引 (SOP)</div>
            <div style="font-size:0.85rem; color:var(--text-secondary); line-height:1.6; margin-top:8px;">
                <p><b>• 步骤一（信息采集）</b>：在常规体检或门诊问卷中询问老年人的静态爱好、社区设施满意度、家庭社会支持、既往外伤史，并结合血常规单据读取血红蛋白值与测量BMI；</p>
                <p><b>• 步骤二（智能打分）</b>：打开本网页系统或使用列线图单项标尺，输入患者数据，1秒内自动计算发病几率并得出风险层级；</p>
                <p><b>• 步骤三（双轨决策）</b>：若总分 ≥ 270 分，启动黄色预警宣教；若总分 ≥ 300 分，启动红色高危签约协议与反事实生活方式调控。</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with g_c2:
        st.markdown("""
        <div class="glass-card">
            <div class="card-header-title">🥋 民族传统适老化微运动方案（八段锦与太极）</div>
            <div style="font-size:0.85rem; color:var(--text-secondary); line-height:1.6; margin-top:8px;">
                <p><b>1. 八段锦'意气形神'和缓训练</b><br>
                国际权威神经科学文献证明，传统八段锦兼具对称平衡、动作舒缓与深呼吸协同特点，能有效刺激海马体脑源性神经营养因子（BDNF）释放，改善SCD老人的双重任务步态平衡，尤其适合有外伤史或跌倒恐惧的老人。</p>
                <p><b>2. 适老化'微运动'（Snack Exercise）策略</b><br>
                针对看电视久坐的习惯，推行'坐四五、动五分'原则。即每坐看电视或打牌45分钟，强制起立室内慢步踱步、活动肩颈或靠墙静蹲3~5分钟，有效打断代谢停滞。</p>
                <p><b>3. 社区同伴支持与社群效应</b><br>
                依托社区居委会与老年协会，建立以楼栋为单位的互助健步小组。通过同伴激励破除独居与低社交频率障碍，形成长效健康行为定势。</p>
            </div>
        </div>
        <div class="glass-card">
            <div class="card-header-title">📖 课题科研档案与文献指引</div>
            <div style="font-size:0.82rem; color:var(--text-secondary); line-height:1.6; margin-top:8px;">
                <p>• <b>研究人群</b>：江西省赣州市中心城区 15 个社区前瞻性 SCD 队列 (N=385)</p>
                <p>• <b>规范标准</b>：TRIPOD+AI 2024 / PROBAST / Riley 2020 现代预测模型准则</p>
                <p>• <b>核心算法</b>：XGBoost 冠军机器学习算法 (Test AUC 0.832, 灵敏度 75.61%, F1 0.7126)</p>
                <p>• <b>学术专著</b>：《社区主观认知下降老年人身体活动不足风险预测模型构建与多算法验证报告》</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem; color:var(--text-muted); font-size:0.78rem;">
    © 2026 赣南医科大学社区老年健康智能医学研究中心 · SCD-ActiPredict AI Platform v2.4 Pro<br>
    All Rights Reserved · Powered by Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
