"""
SCD-ActiPredict AI v2.0 - Community SCD Older Adults Physical Inactivity Risk Prediction System
Production-Grade Streamlit Web Application (Curated 6 Core Independent Clinical Features)
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
# Page Configuration & Metadata
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SCD-ActiPredict AI | 社区SCD老年人活动不足风险预测系统",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------------------------
# High-End Design System (Apple / Vercel / Stripe Aesthetic CSS)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --font-main: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        --bg-body: #F8FAFC;
        --bg-card: #FFFFFF;
        --bg-subtle: #F1F5F9;
        --border-subtle: #E2E8F0;
        --border-focus: #3B82F6;
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --text-muted: #94A3B8;
        --accent-blue: #2563EB;
        --accent-indigo: #4F46E5;
        --accent-purple: #7C3AED;
        --emerald: #10B981;
        --emerald-bg: #ECFDF5;
        --amber: #F59E0B;
        --amber-bg: #FFFBEB;
        --crimson: #EF4444;
        --crimson-bg: #FEF2F2;
        --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.04);
    }

    html, body, [class*="css"] {
        font-family: var(--font-main);
        color: var(--text-primary);
    }

    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3.5rem;
        max-width: 1400px;
    }

    /* Style Streamlit Containers as Cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle) !important;
        border-radius: 16px !important;
        box-shadow: var(--shadow-sm);
        padding: 4px 6px;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: var(--shadow-md);
        border-color: rgba(37, 99, 235, 0.25) !important;
    }

    /* Top Floating Header */
    .hero-navbar {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.06) 0%, rgba(124, 58, 237, 0.06) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--border-subtle);
        border-radius: 18px;
        padding: 18px 24px;
        margin-bottom: 20px;
        box-shadow: var(--shadow-sm);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 15px;
    }

    .brand-title {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 50%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 3px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .brand-subtitle {
        font-size: 0.84rem;
        color: var(--text-secondary);
        font-weight: 500;
    }

    /* Badges & Pills */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        line-height: 1.2;
    }

    .badge-champion {
        background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%);
        color: white;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25);
    }

    .badge-verified {
        background: var(--emerald-bg);
        color: #065F46;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .badge-normal {
        background: var(--emerald-bg);
        color: #047857;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
    }

    .badge-warn {
        background: var(--amber-bg);
        color: #B45309;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
    }

    .badge-danger {
        background: var(--crimson-bg);
        color: #B91C1C;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
    }

    .card-title-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
        padding-bottom: 10px;
        border-bottom: 1px solid var(--border-subtle);
    }

    .card-title-text {
        font-size: 1.02rem;
        font-weight: 700;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .card-desc {
        font-size: 0.80rem;
        color: var(--text-muted);
        margin-top: -6px;
        margin-bottom: 14px;
    }

    /* Preset Selection Bar */
    .preset-header {
        font-size: 0.88rem;
        font-weight: 700;
        color: var(--text-secondary);
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Scoreboard Metrics */
    .metric-score-box {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.05) 0%, rgba(79, 70, 229, 0.08) 100%);
        border: 1px solid rgba(37, 99, 235, 0.2);
        border-radius: 14px;
        padding: 14px 16px;
        text-align: center;
        margin-bottom: 14px;
    }

    .metric-score-val {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin-bottom: 3px;
    }

    .metric-score-lbl {
        font-size: 0.78rem;
        color: var(--text-secondary);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Points breakdown chips */
    .points-chip {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 7px 11px;
        background: var(--bg-subtle);
        border-radius: 8px;
        margin-bottom: 6px;
        font-size: 0.82rem;
        font-weight: 500;
    }

    .points-val {
        font-weight: 700;
        color: var(--accent-blue);
        font-family: monospace;
        font-size: 0.88rem;
    }

    /* Prescription Card */
    .rx-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.06) 0%, rgba(6, 182, 212, 0.06) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 14px;
        padding: 14px 16px;
        margin-top: 14px;
    }

    .rx-title {
        font-size: 0.90rem;
        font-weight: 700;
        color: #047857;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .rx-text {
        font-size: 0.82rem;
        color: var(--text-secondary);
        line-height: 1.55;
    }

    /* Custom Streamlit Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: var(--bg-subtle);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid var(--border-subtle);
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px;
        font-size: 0.88rem;
        font-weight: 600;
        color: var(--text-secondary);
        padding: 0 18px;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--bg-card) !important;
        color: var(--accent-blue) !important;
        box-shadow: var(--shadow-sm);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Mathematical Engine & 6-Feature Prediction Architecture
# ---------------------------------------------------------------------------
FEATURES_6 = [
    'Sedentary_Hobby_Non_Intellectual',
    'Trauma_History',
    'BMI',
    'Age',
    'Hemoglobin',
    'Monthly_Income'
]

# Exact Statsmodels Multivariable Logistic Regression Parameters
LOGIT_PARAMS_6 = {
    'const': -5.1264,
    'Sedentary_Hobby_Non_Intellectual': 2.0977, # aOR = 8.15, P < 0.0001 (Risk)
    'Trauma_History': 1.9599,                  # aOR = 7.10, P < 0.0001 (Risk)
    'BMI': 0.2330,                             # aOR = 1.26, P = 0.0063 (Risk)
    'Age': 0.0736,                             # aOR = 1.08, P = 0.0202 (Risk)
    'Hemoglobin': -0.0460,                      # aOR = 0.95, P = 0.0082 (Protective)
    'Monthly_Income': -0.5338                  # aOR = 0.59, P = 0.0017 (Protective)
}

# Bedside Nomogram 100-Point Geometric Scale Alignment Metrics (MAX_EFFECT_SPAN = BMI span: 2.7959)
MAX_SPAN = 2.7959

@st.cache_resource
def load_6feature_models():
    """Loads 6-feature trained XGBoost model and scaler if available."""
    xgb_path = os.path.join("models", "best_model_6features.pkl")
    scaler_path = os.path.join("models", "standard_scaler_6features.pkl")
    
    if os.path.exists(xgb_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(xgb_path)
            scaler = joblib.load(scaler_path)
            return model, scaler, True
        except Exception:
            pass
    return None, None, False

xgb_model_6, scaler_6, xgb_loaded = load_6feature_models()

def compute_nomogram_points(patient):
    """Calculates exact bedside clinical Nomogram points (0 ~ 450 total scale)."""
    bmi_val = np.clip(patient['BMI'], 18.0, 30.0)
    pts_bmi = 0.2330 * (bmi_val - 18.0) / MAX_SPAN * 100.0

    pts_hobby = 75.028 * float(patient['Sedentary_Hobby_Non_Intellectual'])
    pts_trauma = 70.101 * float(patient['Trauma_History'])

    age_val = np.clip(patient['Age'], 60.0, 85.0)
    pts_age = 0.0736 * (age_val - 60.0) / MAX_SPAN * 100.0

    hb_val = np.clip(patient['Hemoglobin'], 100.0, 160.0)
    pts_hb = 0.0460 * (160.0 - hb_val) / MAX_SPAN * 100.0

    inc_val = np.clip(patient['Monthly_Income'], 0.0, 4.0)
    pts_inc = 0.5338 * (4.0 - inc_val) / MAX_SPAN * 100.0

    total_pts = pts_bmi + pts_hobby + pts_trauma + pts_age + pts_hb + pts_inc
    
    breakdown = {
        'BMI': (pts_bmi, 100.0, "体质指数 (BMI)", "危险因素"),
        'Sedentary_Hobby_Non_Intellectual': (pts_hobby, 75.0, "静态非益智爱好", "危险因素"),
        'Trauma_History': (pts_trauma, 70.1, "跌倒外伤病史", "危险因素"),
        'Age': (pts_age, 65.8, "高龄增长", "危险因素"),
        'Hemoglobin': (pts_hb, 98.7, "血红蛋白水平", "保护因素"),
        'Monthly_Income': (pts_inc, 76.4, "人均月收入", "保护因素"),
    }
    return total_pts, breakdown

def predict_physical_inactivity(patient):
    """Computes predicted probability of physical inactivity using 100% exact math logit closed loop."""
    lp = LOGIT_PARAMS_6['const']
    lp += LOGIT_PARAMS_6['Sedentary_Hobby_Non_Intellectual'] * patient['Sedentary_Hobby_Non_Intellectual']
    lp += LOGIT_PARAMS_6['Trauma_History'] * patient['Trauma_History']
    lp += LOGIT_PARAMS_6['BMI'] * patient['BMI']
    lp += LOGIT_PARAMS_6['Age'] * patient['Age']
    lp += LOGIT_PARAMS_6['Hemoglobin'] * patient['Hemoglobin']
    lp += LOGIT_PARAMS_6['Monthly_Income'] * patient['Monthly_Income']

    prob = 1.0 / (1.0 + np.exp(-lp))
    return float(prob), "临床列线图标准模型 (100% 绝对数学对准)"

# ---------------------------------------------------------------------------
# Hero Navbar
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-navbar">
    <div>
        <div class="brand-title">🏃‍♂️ SCD-ActiPredict AI <span style="font-size:0.85rem; font-weight:600; padding:2px 8px; border-radius:6px; background:rgba(37,99,235,0.1); color:#2563EB;">v2.0 纯6特征极简版</span></div>
        <div class="brand-subtitle">社区主观认知下降（SCD）老年人身体活动不足风险智能预测与临床决策支持系统</div>
    </div>
    <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
        <span class="badge-pill badge-champion">⚡ 6项核心独立特征闭环</span>
        <span class="badge-pill badge-verified">🎯 临床列线图 100% 对齐</span>
        <a href="https://github.com/hhini/old_anactived" target="_blank" style="text-decoration:none;">
            <span class="badge-pill" style="background:#F1F5F9; color:#334155; border:1px solid #CBD5E1;">⭐ GitHub 源码</span>
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tab Navigation
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🧮 6项指标临床智能风险计算器",
    "🏆 7大算法全景评测排行榜",
    "🔬 6项核心特征可解释性与床旁列线图",
    "📘 社区分级干预与适老微运动 SOP"
])

# ===========================================================================
# TAB 1: 6-FEATURE CLINICAL PREDICTION CALCULATOR
# ===========================================================================
with tab1:
    st.markdown('<div class="preset-header">⚡ 典型临床病例一键快速装填 (Quick Presets)</div>', unsafe_allow_html=True)
    
    pre_col1, pre_col2, pre_col3, pre_col4 = st.columns(4)
    
    with pre_col1:
        if st.button("🟢 典型低危老人 (活力健康标杆)", use_container_width=True, help="65岁, BMI 21.0, 无外伤, 无静态爱好, 月入≥4000, Hb 150"):
            st.session_state.calc_age = 65
            st.session_state.calc_bmi = 21.0
            st.session_state.calc_hobby = 0
            st.session_state.calc_trauma = 0
            st.session_state.calc_hb = 150.0
            st.session_state.calc_income = 4

    with pre_col2:
        if st.button("🟡 普查预警老人 (临界筛查切点)", use_container_width=True, help="70岁, BMI 23.8, 偶有跌倒史, 无静态爱好, 月入2000~, Hb 128"):
            st.session_state.calc_age = 70
            st.session_state.calc_bmi = 23.8
            st.session_state.calc_hobby = 0
            st.session_state.calc_trauma = 1
            st.session_state.calc_hb = 128.0
            st.session_state.calc_income = 2

    with pre_col3:
        if st.button("🟠 专科干预老人 (高危共病衰弱)", use_container_width=True, help="76岁, BMI 26.5, 有跌倒外伤, 长期看电视久坐, 月入1000~, Hb 118"):
            st.session_state.calc_age = 76
            st.session_state.calc_bmi = 26.5
            st.session_state.calc_hobby = 1
            st.session_state.calc_trauma = 1
            st.session_state.calc_hb = 118.0
            st.session_state.calc_income = 1

    with pre_col4:
        if st.button("🔴 重度危机老人 (极高危久坐瘫痪)", use_container_width=True, help="82岁, BMI 28.5, 有外伤骨折, 静态爱好, 月入<1000, 贫血 108"):
            st.session_state.calc_age = 82
            st.session_state.calc_bmi = 28.5
            st.session_state.calc_hobby = 1
            st.session_state.calc_trauma = 1
            st.session_state.calc_hb = 108.0
            st.session_state.calc_income = 0

    # Ensure session state defaults
    if "calc_age" not in st.session_state:
        st.session_state.calc_age = 72
        st.session_state.calc_bmi = 24.5
        st.session_state.calc_hobby = 1
        st.session_state.calc_trauma = 1
        st.session_state.calc_hb = 122.0
        st.session_state.calc_income = 2

    st.write("") # spacing

    # 2-Column Split: Inputs (6 Features) on Left, Results Dashboard on Right
    left_col, right_col = st.columns([6, 6], gap="large")

    with left_col:
        # Card 1: 4 Independent Risk Factors (Encapsulated in native bordered container)
        with st.container(border=True):
            st.markdown("""
            <div class="card-title-bar">
                <div class="card-title-text">🚨 4大独立致病危险因素 (Risk Factors)</div>
                <span class="badge-danger">正向推高发病对数几率</span>
            </div>
            <div class="card-desc">以下4项指标每上升一个层级，均显著放大老年人身体活动不足的发生几率</div>
            """, unsafe_allow_html=True)

            # 1. Sedentary Hobby
            hobby_labels = {
                0: "否 · 无静态不良爱好（平时喜爱散步/太极/家务等）",
                1: "是 · 存在长期看电视/久坐闲聊等静态非益智爱好"
            }
            val_hobby = st.radio(
                "1. 静态非益智类休闲爱好 (全模型最强独立致病因素, aOR=8.15)",
                options=[0, 1],
                format_func=lambda x: hobby_labels[x],
                index=int(st.session_state.calc_hobby),
                help="低脑耗与低能耗娱乐严重挤占活动物理时间窗口，钝化多巴胺神经犒赏，赋分高达 75.0 分！"
            )

            st.write("")

            # 2. Trauma History
            trauma_labels = {
                0: "无 · 既往无严重跌倒外伤或骨折史",
                1: "有 · 曾发生跌倒外伤、骨折或关节扭伤病史"
            }
            val_trauma = st.radio(
                "2. 既往外伤骨折或跌倒病史 (aOR=7.10)",
                options=[0, 1],
                format_func=lambda x: trauma_labels[x],
                index=int(st.session_state.calc_trauma),
                help="外伤遗留慢性关节酸痛并引发强烈的'跌倒恐惧综合征'（Fear of Falling），赋分 70.1 分！"
            )

            st.write("")

            # 3. BMI with dynamic badge
            c_bmi1, c_bmi2 = st.columns([8, 4])
            with c_bmi1:
                val_bmi = st.slider(
                    "3. 体质指数 BMI (kg/m², aOR=1.26)",
                    min_value=18.0,
                    max_value=30.0,
                    value=float(st.session_state.calc_bmi),
                    step=0.1,
                    help="体质指数每增加1单位，活动不足几率递增26.2%。BMI=30时达到100分基准满分！"
                )
            with c_bmi2:
                st.write("")
                st.write("")
                if val_bmi < 24.0:
                    st.markdown(f'<span class="badge-normal">标准体型 ({val_bmi:.1f})</span>', unsafe_allow_html=True)
                elif val_bmi < 28.0:
                    st.markdown(f'<span class="badge-warn">超重前期 ({val_bmi:.1f})</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="badge-danger">肥胖高危 ({val_bmi:.1f})</span>', unsafe_allow_html=True)

            st.write("")

            # 4. Age
            c_age1, c_age2 = st.columns([8, 4])
            with c_age1:
                val_age = st.slider(
                    "4. 年龄（周岁, aOR=1.08）",
                    min_value=60,
                    max_value=85,
                    value=int(st.session_state.calc_age),
                    step=1,
                    help="高龄伴随骨骼肌退行性衰退与心肺功能储备下降，跨度赋分最高达 65.8 分。"
                )
            with c_age2:
                st.write("")
                st.write("")
                if val_age < 70:
                    st.markdown(f'<span class="badge-normal">低龄老年 ({val_age}岁)</span>', unsafe_allow_html=True)
                elif val_age < 80:
                    st.markdown(f'<span class="badge-warn">中龄老年 ({val_age}岁)</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="badge-danger">高龄脆弱 ({val_age}岁)</span>', unsafe_allow_html=True)

        st.write("")

        # Card 2: 2 Independent Protective Factors (Encapsulated in native bordered container)
        with st.container(border=True):
            st.markdown("""
            <div class="card-title-bar">
                <div class="card-title-text">🛡️ 2大独立保护抵御因素 (Protective Buffers)</div>
                <span class="badge-verified">负向抵消发病风险</span>
            </div>
            <div class="card-desc">以下2项指标水平越高，对身体活动不足的抵御保护能力越强，显著压降总评分</div>
            """, unsafe_allow_html=True)

            # 5. Hemoglobin
            c_hb1, c_hb2 = st.columns([8, 4])
            with c_hb1:
                val_hb = st.slider(
                    "5. 血红蛋白浓度 (g/L, aOR=0.95)",
                    min_value=100.0,
                    max_value=160.0,
                    value=float(st.session_state.calc_hb),
                    step=1.0,
                    help="有氧耐力与机体组织携氧底座。160 g/L为0分安全基准；低于120 g/L提示贫血，赋分高达70~98分！"
                )
            with c_hb2:
                st.write("")
                st.write("")
                if val_hb >= 130.0:
                    st.markdown(f'<span class="badge-normal">耐力充沛 ({val_hb:.0f} g/L)</span>', unsafe_allow_html=True)
                elif val_hb >= 120.0:
                    st.markdown(f'<span class="badge-warn">临界安全 ({val_hb:.0f} g/L)</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="badge-danger">贫血乏力 ({val_hb:.0f} g/L)</span>', unsafe_allow_html=True)

            st.write("")

            # 6. Monthly Income
            inc_map = {
                0: "< 1000 元/月 (赋分 76.4 分)",
                1: "1000 ~ 1999 元/月 (赋分 57.3 分)",
                2: "2000 ~ 2999 元/月 (赋分 38.2 分)",
                3: "3000 ~ 3999 元/月 (赋分 19.1 分)",
                4: "≥ 4000 元/月 (0分基准标杆)"
            }
            val_inc = st.selectbox(
                "6. 家庭人均月收入阶梯 (aOR=0.59)",
                options=[0, 1, 2, 3, 4],
                format_func=lambda x: inc_map[x],
                index=int(st.session_state.calc_income),
                help="经济充裕度显著增强文体消费、健康食品摄入及营养保障能力，每提升一档保护比值比达0.59。"
            )

    # Current Patient Dict
    current_patient = {
        'Sedentary_Hobby_Non_Intellectual': float(val_hobby),
        'Trauma_History': float(val_trauma),
        'BMI': float(val_bmi),
        'Age': float(val_age),
        'Hemoglobin': float(val_hb),
        'Monthly_Income': float(val_inc)
    }

    # Execute Prediction and Bedside Nomogram Points Calculation
    total_pts, pts_breakdown = compute_nomogram_points(current_patient)
    risk_prob, engine_name = predict_physical_inactivity(current_patient)
    risk_pct = risk_prob * 100.0

    # Risk Tier Stratification
    if risk_prob < 0.20:
        tier_title = "🟢 低危梯队 (Low Risk Tier)"
        tier_color = "#10B981"
        tier_bg = "rgba(16, 185, 129, 0.12)"
        clinical_advice = "身体活动储备极佳，暂无紧迫活动不足风险。建议维持当前健康作息，常规开展年度社区健康宣教与常规体能复查。"
    elif risk_prob < 0.50:
        tier_title = "🟡 普查预警梯队 (Screening Alert Tier)"
        tier_color = "#F59E0B"
        tier_bg = "rgba(245, 158, 11, 0.12)"
        clinical_advice = "已突破社区普查筛查预警切点（P ≥ 0.270）。存在隐匿性久坐倾向，建议社区家庭医生纳入重点慢病运动建档，发放微运动处方打卡督导。"
    else:
        tier_title = "🔴 专科强化干预梯队 (Clinical High Risk Tier)"
        tier_color = "#EF4444"
        tier_bg = "rgba(239, 68, 68, 0.12)"
        clinical_advice = "已达专科强化干预切点（P ≥ 0.500），处于严重的衰弱与重度活动不足危险期！需立即启动三甲专科与全科医生一对一多学科联合靶向干预。"

    # Right Column: Visual Dashboard
    with right_col:
        with st.container(border=True):
            st.markdown(f"""
            <div class="card-title-bar" style="border-top: 3px solid {tier_color}; padding-top:6px;">
                <div class="card-title-text">📊 预测诊断与床旁量化中枢</div>
                <span class="badge-pill" style="background:{tier_bg}; color:{tier_color}; font-weight:700;">{tier_title}</span>
            </div>
            <div style="text-align:center; font-size:0.86rem; font-weight:600; color:#64748B; margin-bottom:-5px;">
                身体活动不足发生概率预测 (Predicted Risk Probability)
            </div>
            """, unsafe_allow_html=True)

            # Plotly Semi-Circular Risk Gauge (Zero collision, clean spacing)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_pct,
                domain={'x': [0.08, 0.92], 'y': [0.02, 0.98]},
                number={'suffix': "%", 'font': {'size': 38, 'color': tier_color, 'family': "Plus Jakarta Sans, sans-serif"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1.2, 'tickcolor': "#94A3B8", 'tickmode': "array", 'tickvals': [0, 20, 27, 50, 75, 100]},
                    'bar': {'color': tier_color, 'thickness': 0.32},
                    'bgcolor': "rgba(0,0,0,0.03)",
                    'borderwidth': 1,
                    'bordercolor': "#E2E8F0",
                    'steps': [
                        {'range': [0, 20], 'color': 'rgba(16, 185, 129, 0.18)'},
                        {'range': [20, 50], 'color': 'rgba(245, 158, 11, 0.18)'},
                        {'range': [50, 100], 'color': 'rgba(239, 68, 68, 0.18)'}
                    ],
                    'threshold': {
                        'line': {'color': "#DC2626", 'width': 3.5},
                        'thickness': 0.85,
                        'value': 50.0
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=200,
                margin=dict(l=15, r=15, t=10, b=5)
            )
            st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

            # Scorebox & Nomogram Points
            m_c1, m_c2 = st.columns(2)
            with m_c1:
                st.markdown(f"""
                <div class="metric-score-box">
                    <div class="metric-score-val" style="color:{tier_color};">{total_pts:.1f}</div>
                    <div class="metric-score-lbl">临床列线图总评分 (Total Points)</div>
                </div>
                """, unsafe_allow_html=True)
            with m_c2:
                st.markdown(f"""
                <div class="metric-score-box">
                    <div class="metric-score-val" style="color:var(--accent-blue);">{risk_prob:.3f}</div>
                    <div class="metric-score-lbl">Sigmoid 发病概率 P</div>
                </div>
                """, unsafe_allow_html=True)

            # Bedside Points Detail Breakdown
            st.markdown("<b>📋 6项核心指标床旁得分明细 (Bedside Points Breakdown):</b>", unsafe_allow_html=True)
            for code, (pts, max_pts, name_cn, role) in pts_breakdown.items():
                color_badge = "color:#EF4444;" if role == "危险因素" else "color:#2563EB;"
                st.markdown(f"""
                <div class="points-chip">
                    <span>{name_cn} <span style="{color_badge} font-size:0.75rem;">({role})</span></span>
                    <span class="points-val">+{pts:.1f} 分 <span style="color:#94A3B8; font-size:0.75rem;">/ 满分{max_pts:.1f}</span></span>
                </div>
                """, unsafe_allow_html=True)

            # Prescription Card
            st.markdown(f"""
            <div class="rx-card">
                <div class="rx-title">🩺 循证医学个体化临床决策建议</div>
                <div class="rx-text">{clinical_advice}</div>
            </div>
            """, unsafe_allow_html=True)

    # Secondary Visual: Dynamic Waterfall / Impact Analysis
    st.write("")
    with st.container(border=True):
        st.markdown("""
        <div class="card-title-bar">
            <div class="card-title-text">🧬 6大特征个体边际贡献度分析 (Patient SHAP Attribution)</div>
            <span class="badge-pill" style="background:#EEF2FF; color:#4F46E5;">对数几率贡献量解析</span>
        </div>
        """, unsafe_allow_html=True)

        wf_cols = st.columns([7, 5])
        with wf_cols[0]:
            deltas = []
            ref_means = {'Sedentary_Hobby_Non_Intellectual': 0.257, 'Trauma_History': 0.175, 'BMI': 22.18, 'Age': 70.29, 'Hemoglobin': 127.0, 'Monthly_Income': 1.92}
            for f in FEATURES_6:
                coef = LOGIT_PARAMS_6[f]
                d = (current_patient[f] - ref_means[f]) * coef
                deltas.append({
                    'Feature': pts_breakdown[f][2],
                    'Contribution': d,
                    'Direction': "推升风险 (Risk +)" if d > 0 else "降低风险 (Buffer -)",
                    'Color': "#EF4444" if d > 0 else "#2563EB"
                })
            df_wf = pd.DataFrame(deltas).sort_values(by='Contribution', ascending=True)

            fig_bar = go.Figure(go.Bar(
                x=df_wf['Contribution'],
                y=df_wf['Feature'],
                orientation='h',
                marker=dict(color=df_wf['Color']),
                text=[f"{v:+.2f}" for v in df_wf['Contribution']],
                textposition='auto'
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(title="对数几率 (Log-Odds) 偏移量", zeroline=True, zerolinewidth=1.5, zerolinecolor='#64748B'),
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

        with wf_cols[1]:
            st.markdown("""
            <div style="font-size:0.84rem; color:var(--text-secondary); line-height:1.6; padding:12px 16px; background:var(--bg-subtle); border-radius:12px; margin-top:10px;">
                <b>💡 临床归因解读指南：</b><br/>
                • <span style="color:#EF4444; font-weight:700;">红色横条</span>：代表该项指标高于群体基线，正在显著<b>推高</b>该老年人的活动不足概率；<br/>
                • <span style="color:#2563EB; font-weight:700;">蓝色横条</span>：代表该项保护指标发挥正面抗阻作用，正在显著<b>拉低</b>发病风险；<br/>
                • <b>最高杠杆干预靶点</b>：若存在静态非益智爱好，只要指导老人以八段锦或散步替代久坐，即可立减 <b>75.0 分</b>，瞬间将预测风险压降 <b>30%~45%</b>！
            </div>
            """, unsafe_allow_html=True)


# ===========================================================================
# TAB 2: 7 ALGORITHMS LEADERBOARD & ARENA
# ===========================================================================
with tab2:
    with st.container(border=True):
        st.markdown("""
        <div class="card-title-bar">
            <div class="card-title-text">🏆 7大主流机器学习算法独立盲态测试集排行榜 (Benchmark Arena)</div>
            <span class="badge-champion">XGBoost 冠军模型领跑全场</span>
        </div>
        <div class="card-desc">在独立测试集（N=116）上严格遵循 5 折交叉验证与 Optuna 贝叶斯寻优参数进行无偏量化评测</div>
        """, unsafe_allow_html=True)

        df_models_path = os.path.join("results", "08_multimodel_performance_comparison.csv")
        if os.path.exists(df_models_path):
            df_models = pd.read_csv(df_models_path)
            st.dataframe(
                df_models[['Model', 'Test_ROC_AUC', 'Test_AUC_95CI', 'Sensitivity_Recall', 'Specificity', 'PPV_Precision', 'F1_Score', 'Brier_Score']],
                use_container_width=True,
                column_config={
                    "Model": "算法名称",
                    "Test_ROC_AUC": st.column_config.NumberColumn("ROC-AUC", format="%.4f"),
                    "Test_AUC_95CI": "95% 置信区间",
                    "Sensitivity_Recall": st.column_config.NumberColumn("敏感度 (Recall)", format="%.2f%%"),
                    "Specificity": st.column_config.NumberColumn("特异度 (Specificity)", format="%.2f%%"),
                    "PPV_Precision": st.column_config.NumberColumn("精确率 (Precision)", format="%.2f%%"),
                    "F1_Score": st.column_config.NumberColumn("F1 分数", format="%.4f"),
                    "Brier_Score": st.column_config.NumberColumn("Brier 校准分", format="%.4f"),
                },
                hide_index=True
            )

    st.write("")
    g_c1, g_c2 = st.columns(2)
    with g_c1:
        with st.container(border=True):
            st.markdown('<div class="card-title-text" style="margin-bottom:10px;">📈 7大算法测试集 ROC 曲线全景对比 (Fig 4)</div>', unsafe_allow_html=True)
            img_roc = os.path.join("figures", "Fig4_MultiModel_ROC_Curves_Train_and_Test.png")
            if os.path.exists(img_roc):
                st.image(img_roc, use_container_width=True)

    with g_c2:
        with st.container(border=True):
            st.markdown('<div class="card-title-text" style="margin-bottom:10px;">🕸️ 六维性能雷达竞技场对比 (Fig 16)</div>', unsafe_allow_html=True)
            img_radar = os.path.join("figures", "Fig16_MultiModel_Radar_Performance_Comparison.png")
            if os.path.exists(img_radar):
                st.image(img_radar, use_container_width=True)

    st.write("")
    g_c3, g_c4 = st.columns(2)
    with g_c3:
        with st.container(border=True):
            st.markdown('<div class="card-title-text" style="margin-bottom:10px;">🧩 7大算法混淆矩阵大拼图 (Fig 15)</div>', unsafe_allow_html=True)
            img_cm = os.path.join("figures", "Fig15_MultiModel_Confusion_Matrices_Mosaic.png")
            if os.path.exists(img_cm):
                st.image(img_cm, use_container_width=True)

    with g_c4:
        with st.container(border=True):
            st.markdown('<div class="card-title-text" style="margin-bottom:10px;">📊 三阶临床风险分层发病率比 (Fig 17)</div>', unsafe_allow_html=True)
            img_strat = os.path.join("figures", "Fig17_Clinical_Risk_Stratification_and_Outcomes.png")
            if os.path.exists(img_strat):
                st.image(img_strat, use_container_width=True)


# ===========================================================================
# TAB 3: 6 CORE FEATURES XAI & CLINICAL NOMOGRAM
# ===========================================================================
with tab3:
    with st.container(border=True):
        st.markdown("""
        <div class="card-title-bar">
            <div class="card-title-text">📐 临床床旁诊断列线图系统 (Clinical Nomogram)</div>
            <span class="badge-verified">数学精确闭环对准 (0~450分)</span>
        </div>
        <div class="card-desc">由多因素二元Logistic回归方程100%严格几何映射，标尺与发病概率完全闭环</div>
        """, unsafe_allow_html=True)
        img_nomo = os.path.join("figures", "Fig11_Clinical_Diagnostic_Nomogram.png")
        if os.path.exists(img_nomo):
            st.image(img_nomo, use_container_width=True)

    st.write("")
    x_c1, x_c2 = st.columns(2)
    with x_c1:
        with st.container(border=True):
            st.markdown('<div class="card-title-text" style="margin-bottom:10px;">🌲 6项核心特征精选森林图 (Fig 3B)</div>', unsafe_allow_html=True)
            img_forest = os.path.join("figures", "Fig3B_Forest_Plot_Screened_6_Independent_Factors.png")
            if os.path.exists(img_forest):
                st.image(img_forest, use_container_width=True)

    with x_c2:
        with st.container(border=True):
            st.markdown('<div class="card-title-text" style="margin-bottom:10px;">🐝 6项核心特征全局 SHAP 重要度与蜂群图 (Fig 8B)</div>', unsafe_allow_html=True)
            img_shap6 = os.path.join("figures", "Fig8B_SHAP_Global_Summary_and_Beeswarm_Screened_6_Factors.png")
            if os.path.exists(img_shap6):
                st.image(img_shap6, use_container_width=True)


# ===========================================================================
# TAB 4: INTERVENTION GUIDELINES & SENIOR-FRIENDLY MICRO-EXERCISE SOP
# ===========================================================================
with tab4:
    with st.container(border=True):
        st.markdown("""
        <div class="card-title-bar">
            <div class="card-title-text">📘 社区SCD老年人“防-筛-控”三级分层干预网络</div>
            <span class="badge-champion">健康老龄化国家战略闭环</span>
        </div>
        """, unsafe_allow_html=True)

        s_c1, s_c2, s_c3 = st.columns(3)
        with s_c1:
            st.markdown("""
            <div style="background:var(--emerald-bg); border:1px solid rgba(16,185,129,0.3); border-radius:12px; padding:16px;">
                <div style="color:#065F46; font-weight:700; font-size:0.95rem; margin-bottom:8px;">一级预防：低危群体常规宣教</div>
                <div style="font-size:0.80rem; color:#047857; line-height:1.6;">
                    • <b>适用对象</b>：预测风险 &lt; 20% 老年人；<br/>
                    • <b>核心目标</b>：巩固良好生活方式，阻断衰弱发生；<br/>
                    • <b>干预路径</b>：社区健康大讲堂普及适度有氧运动益处，发放《活力老人微运动指南》，每年常规进行一次体能与认知随访。
                </div>
            </div>
            """, unsafe_allow_html=True)

        with s_c2:
            st.markdown("""
            <div style="background:var(--amber-bg); border:1px solid rgba(245,158,11,0.3); border-radius:12px; padding:16px;">
                <div style="color:#B45309; font-weight:700; font-size:0.95rem; margin-bottom:8px;">二级预防：临界预警社区微干预</div>
                <div style="font-size:0.80rem; color:#92400E; line-height:1.6;">
                    • <b>适用对象</b>：风险 20% ~ 50%（切点 P ≥ 0.270）；<br/>
                    • <b>核心目标</b>：逆转久坐倾向，防范跌倒复发；<br/>
                    • <b>干预路径</b>：家庭医生签约建档，推行“坐45动5分”零食运动法则，动员参与社区太极或八段锦团队，实施同伴结对打卡。
                </div>
            </div>
            """, unsafe_allow_html=True)

        with s_c3:
            st.markdown("""
            <div style="background:var(--crimson-bg); border:1px solid rgba(239,68,68,0.3); border-radius:12px; padding:16px;">
                <div style="color:#991B1B; font-weight:700; font-size:0.95rem; margin-bottom:8px;">三级预防：高危重度专科靶向</div>
                <div style="font-size:0.80rem; color:#B91C1C; line-height:1.6;">
                    • <b>适用对象</b>：预测风险 ≥ 50% 极重高危老年人；<br/>
                    • <b>核心目标</b>：阻断认知迅速衰退与重度失能；<br/>
                    • <b>干预路径</b>：转诊三甲医院老年科/康复科，全面评估心肺与下肢肌力，定制抗阻微训练，纠正贫血与营养不良，家庭环境无障碍改造。
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.markdown("""
        <div style="background:var(--bg-card); border:1px solid var(--border-subtle); border-radius:12px; padding:18px;">
            <div style="font-weight:700; font-size:0.98rem; margin-bottom:8px; color:var(--text-primary);">🥋 适老化微运动（Exercise Snacks）核心动作处方：</div>
            <div style="font-size:0.83rem; color:var(--text-secondary); line-height:1.65;">
                1. <b>扶椅微蹲（Chair Squat）</b>：双手轻扶椅背，双脚与肩同宽，慢速下蹲至大腿与地面呈45度，每组8次，每日2组。强健股四头肌，增强防跌倒平衡力；<br/>
                2. <b>坐姿提踵（Seated Calf Raise）</b>：端坐座椅前三分之一，双足脚尖着地，用力抬高脚后跟维持3秒后缓慢放下，每组15次。激活“人体第二心脏”小腿比目鱼肌，加速血液回流；<br/>
                3. <b>八段锦·两手托天理三焦</b>：晨起或久坐后配合深长呼吸双手上托，拉伸胸廓与脊柱，促进脑部供血供氧。
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div style="text-align:center; padding-top:25px; font-size:0.78rem; color:#94A3B8;">
    SCD-ActiPredict AI v2.0 · 临床级 6 项核心独立特征决策系统 · 基于 TRIPOD+AI 2024 指南规范设计<br/>
    仅供科研流调、临床辅助决策与教学演示使用 · 不可替代专业执业医师现场诊断
</div>
""", unsafe_allow_html=True)
