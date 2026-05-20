import datetime as dt
import pandas as pd
import streamlit as st

from model import WaterInput, predict

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="WaterFlow — Potability",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom CSS — Aquatic + data-dashboard hybrid
# =============================================================================
CSS = """
<style>
.stApp {
    background:
        radial-gradient(1000px 600px at 100% 0%, #d8eef6 0%, transparent 60%),
        radial-gradient(800px 500px at 0% 100%, #c7e6f5 0%, transparent 55%),
        #f4f9fc;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b3556 0%, #0a4a7a 50%, #0c5d96 100%);
}
section[data-testid="stSidebar"] * { color: #e6f1fb !important; }
section[data-testid="stSidebar"] .stButton button {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.18);
    color: #e6f1fb !important;
    width: 100%;
    text-align: left;
    border-radius: 8px;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255,255,255,0.18);
    border-color: rgba(34,211,238,0.5);
}

/* Hero header */
.wf-hero {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 24px;
    padding-bottom: 8px;
    margin-bottom: 18px;
}
.wf-hero h1 {
    margin: 0;
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -0.025em;
    color: #07223a;
}
.wf-hero p { margin: 4px 0 0; color: #4b6478; font-size: 14px; }
.wf-live-tag {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    background: #fff;
    border: 1px solid #cfe5f0;
    border-radius: 999px;
    font-size: 11px;
    font-family: ui-monospace, 'JetBrains Mono', monospace;
    font-weight: 600;
    color: #0a4a7a;
}
.wf-live-tag::before {
    content: "";
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 10px #10b981;
}

/* Section label */
.wf-section-label {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 8px 0 14px;
}
.wf-section-num {
    font-family: ui-monospace, 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #0a4a7a;
    background: rgba(34,211,238,0.12);
    border: 1px solid rgba(34,211,238,0.3);
    padding: 3px 8px;
    border-radius: 4px;
    font-weight: 700;
}
.wf-section-label h2 {
    margin: 0;
    font-size: 20px;
    font-weight: 700;
    color: #07223a;
    letter-spacing: -0.01em;
}
.wf-src {
    margin-left: auto;
    font-size: 11px;
    color: #64748b;
    font-family: ui-monospace, 'JetBrains Mono', monospace;
}

/* KPI cards (custom st.metric replacement) */
.wf-kpi {
    background: #fff;
    border: 1px solid #d6e6ee;
    border-radius: 16px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    height: 100%;
}
.wf-kpi::before {
    content: "";
    position: absolute;
    top: 0; right: 0;
    width: 80px; height: 80px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(34,211,238,0.12) 0%, transparent 70%);
}
.wf-kpi .icon {
    width: 36px; height: 36px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    color: #fff;
    font-size: 18px;
    margin-bottom: 14px;
}
.wf-kpi .icon.blue   { background: linear-gradient(135deg, #2aa9ff 0%, #0a4a7a 100%); }
.wf-kpi .icon.warn   { background: linear-gradient(135deg, #f59e0b 0%, #b45309 100%); }
.wf-kpi .icon.danger { background: linear-gradient(135deg, #ef4444 0%, #991b1b 100%); }
.wf-kpi .icon.good   { background: linear-gradient(135deg, #10b981 0%, #047857 100%); }
.wf-kpi .value {
    font-family: ui-monospace, 'JetBrains Mono', monospace;
    font-size: 32px;
    font-weight: 700;
    color: #07223a;
    line-height: 1;
    letter-spacing: -0.02em;
}
.wf-kpi .value .unit { font-size: 14px; color: #64748b; font-weight: 500; margin-left: 4px; }
.wf-kpi .label { font-size: 12px; color: #4b6478; margin-top: 8px; line-height: 1.5; }
.wf-kpi .delta {
    margin-top: 10px;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    font-family: ui-monospace, monospace;
    color: #16a34a;
    background: #dcfce7;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
}
.wf-kpi .delta.bad { color: #b91c1c; background: #fee2e2; }

/* Generic content card */
.wf-card {
    background: #fff;
    border: 1px solid #d6e6ee;
    border-radius: 16px;
    padding: 18px 22px;
}
.wf-card .ttl { font-size: 13px; font-weight: 700; color: #07223a; margin-bottom: 4px; }
.wf-card .sub { font-size: 11px; color: #64748b; margin-bottom: 12px; font-family: ui-monospace, monospace; }

/* Predictor header */
.wf-pred-header {
    display: flex; align-items: center; gap: 16px; margin-bottom: 8px;
}
.wf-pred-header .h-logo {
    width: 54px; height: 54px;
    border-radius: 16px;
    background: linear-gradient(135deg, #2aa9ff 0%, #0a4a7a 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 28px;
    color: #fff;
    box-shadow: 0 10px 30px -10px rgba(10,74,122,0.55);
}
.wf-pred-header h2 { margin: 0; font-size: 28px; font-weight: 800; letter-spacing: -0.02em; color: #07223a; }
.wf-pred-header p { margin: 4px 0 0; font-size: 13px; color: #4b6478; }
.wf-pred-header .sample-id {
    margin-left: auto;
    font-family: ui-monospace, 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #0a4a7a;
    background: rgba(34,211,238,0.08);
    border: 1px solid rgba(34,211,238,0.3);
    padding: 6px 12px;
    border-radius: 6px;
    font-weight: 600;
}

/* Section divider */
.wf-divider {
    display: flex;
    align-items: center;
    gap: 14px;
    color: #94a3b8;
    font-size: 11px;
    font-family: ui-monospace, monospace;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 24px 0 16px;
}
.wf-divider::before, .wf-divider::after {
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, #cbd5e1 50%, transparent 100%);
}

/* Result card */
.wf-result {
    padding: 26px;
    border-radius: 18px;
    color: #fff;
}
.wf-result.safe   { background: linear-gradient(135deg, #16a34a 0%, #0f766e 100%); box-shadow: 0 25px 60px -25px rgba(15,118,110,0.6); }
.wf-result.unsafe { background: linear-gradient(135deg, #dc2626 0%, #7c2d12 100%); box-shadow: 0 25px 60px -25px rgba(220,38,38,0.6); }
.wf-result.idle   { background: linear-gradient(135deg, #475569 0%, #1e293b 100%); }
.wf-result .verdict { font-size: 28px; font-weight: 800; letter-spacing: -0.01em; display: flex; align-items: center; gap: 10px; }
.wf-result .sub { font-size: 12px; opacity: 0.85; margin-top: 2px; font-family: ui-monospace, monospace; }
.wf-result .prob { margin-top: 16px; font-family: ui-monospace, 'JetBrains Mono', monospace; font-size: 56px; font-weight: 700; line-height: 1; }
.wf-result .prob small { font-size: 18px; opacity: 0.65; margin-left: 4px; font-weight: 500; }
.wf-result .progress { margin-top: 16px; height: 8px; border-radius: 999px; background: rgba(255,255,255,0.2); overflow: hidden; }
.wf-result .progress .fill { height: 100%; border-radius: 999px; background: rgba(255,255,255,0.9); }
.wf-result .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 18px; }
.wf-result .stat { padding: 10px 12px; border-radius: 10px; background: rgba(255,255,255,0.14); }
.wf-result .stat .k { font-size: 10px; opacity: 0.85; text-transform: uppercase; letter-spacing: 0.08em; }
.wf-result .stat .v { font-size: 16px; font-weight: 700; margin-top: 2px; font-family: ui-monospace, monospace; }

/* Inputs: customize slider color via accent */
div[data-baseweb="slider"] [role="slider"] { background: #0a4a7a !important; border-color: #0a4a7a !important; }
div[data-testid="stSlider"] label { font-weight: 600; color: #07223a; font-size: 13px; }

/* Primary button (predictor CTA) */
.stButton > button[kind="primary"], .stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #2aa9ff 0%, #0a4a7a 100%);
    color: #fff;
    border: none;
    font-weight: 700;
    padding: 12px 24px;
    border-radius: 12px;
    box-shadow: 0 12px 25px -8px rgba(10,74,122,0.55);
}

/* Did-you-know callout */
.wf-callout {
    padding: 18px 22px;
    border-radius: 16px;
    background: linear-gradient(135deg, #0a4a7a 0%, #0c5d96 100%);
    color: #e6f1fb;
}
.wf-callout .ttl { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.9; }
.wf-callout .body { font-size: 13px; line-height: 1.6; margin-top: 8px; opacity: 0.95; }

/* Hide streamlit chrome */
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* Ideal-range badge */
.wf-ideal { font-size: 10px; padding: 2px 7px; border-radius: 4px; font-weight: 600; margin-left: 6px; }
.wf-ideal.ok   { background: #dcfce7; color: #15803d; }
.wf-ideal.warn { background: #fef3c7; color: #92400e; }
.wf-ideal.bad  { background: #fee2e2; color: #991b1b; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =============================================================================
# Variable metadata
# =============================================================================
META = {
    "ph":              {"label": "pH",                "unit": "",       "min": 0.0, "max": 14.0,   "step": 0.1,  "default": 7.0,    "ideal": (6.5, 8.5),   "group": "chem"},
    "hardness":        {"label": "Dureté",            "unit": "mg/L",   "min": 0.0, "max": 500.0,  "step": 1.0,  "default": 196.0,  "ideal": (60, 180),    "group": "min"},
    "solids":          {"label": "Solides dissous",   "unit": "ppm",    "min": 0.0, "max": 60000., "step": 10.0, "default": 22014., "ideal": (0, 1000),    "group": "min"},
    "chloramines":     {"label": "Chloramines",       "unit": "mg/L",   "min": 0.0, "max": 15.0,   "step": 0.1,  "default": 7.1,    "ideal": (0, 4),       "group": "cont"},
    "sulfate":         {"label": "Sulfate",           "unit": "mg/L",   "min": 0.0, "max": 500.0,  "step": 1.0,  "default": 333.0,  "ideal": (0, 250),     "group": "min"},
    "conductivity":    {"label": "Conductivité",      "unit": "μS/cm",  "min": 0.0, "max": 800.0,  "step": 1.0,  "default": 426.0,  "ideal": (200, 800),   "group": "chem"},
    "organic_carbon":  {"label": "Carbone organique", "unit": "mg/L",   "min": 0.0, "max": 30.0,   "step": 0.1,  "default": 14.3,   "ideal": (0, 4),       "group": "cont"},
    "trihalomethanes": {"label": "Trihalométhanes",   "unit": "μg/L",   "min": 0.0, "max": 120.0,  "step": 0.5,  "default": 66.4,   "ideal": (0, 80),      "group": "cont"},
    "turbidity":       {"label": "Turbidité",         "unit": "NTU",    "min": 0.0, "max": 7.0,    "step": 0.1,  "default": 3.96,   "ideal": (0, 5),       "group": "chem"},
}
GROUPS = {
    "chem": ("⚗", "Chimie",        "pH · conductivité · turbidité"),
    "min":  ("🧂", "Minéraux",      "Dureté · sulfate · solides"),
    "cont": ("⚠", "Contaminants",  "Chloramines · COT · THM"),
}
PRESETS = {
    "💧 Eau potable typique": {"ph": 6.5, "hardness": 150.0, "solids": 350.0, "chloramines": 2.5, "sulfate": 200.0, "conductivity": 420.0, "organic_carbon": 2.5, "trihalomethanes": 40.0, "turbidity": 2.0},
    "⚠️ Eau contaminée":      {"ph": 5.1, "hardness": 280.0, "solids": 35000.0, "chloramines": 9.5, "sulfate": 410.0, "conductivity": 720.0, "organic_carbon": 18.0, "trihalomethanes": 105.0, "turbidity": 6.4},
    "⛲ Eau de source":        {"ph": 6.5, "hardness": 90.0, "solids": 180.0, "chloramines": 0.2, "sulfate": 40.0, "conductivity": 280.0, "organic_carbon": 1.1, "trihalomethanes": 5.0, "turbidity": 0.8},
}

# =============================================================================
# Session state
# =============================================================================
if "inputs" not in st.session_state:
    st.session_state.inputs = {k: m["default"] for k, m in META.items()}
if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None
if "source" not in st.session_state:
    st.session_state.source = None

# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:18px;">
          <div style="width:40px;height:40px;border-radius:12px;
                      background:rgba(34,211,238,0.18);
                      border:1px solid rgba(34,211,238,0.4);
                      display:flex;align-items:center;justify-content:center;font-size:22px;">💧</div>
          <div>
            <div style="font-weight:800;font-size:17px;letter-spacing:-0.01em;">WaterFlow</div>
            <div style="font-size:11px;opacity:0.7;font-family:monospace;">v1.0 · live</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='font-size:10px;text-transform:uppercase;letter-spacing:0.12em;font-weight:700;opacity:0.6;font-family:monospace;'>// Échantillons</div>", unsafe_allow_html=True)
    for name, vals in PRESETS.items():
        if st.button(name, key=f"preset_{name}", use_container_width=True):
            st.session_state.inputs = dict(vals)
            for k, v in vals.items():
                st.session_state[f"slider_{k}"] = float(v)
            st.rerun()

    st.markdown("<div style='font-size:10px;text-transform:uppercase;letter-spacing:0.12em;font-weight:700;opacity:0.6;font-family:monospace;margin-top:10px;'>// Outils</div>", unsafe_allow_html=True)
    if st.button("↻  Réinitialiser", key="reset", use_container_width=True):
        st.session_state.inputs = {k: m["default"] for k, m in META.items()}
        for k, m in META.items():
            st.session_state[f"slider_{k}"] = float(m["default"])
        st.rerun()

    st.markdown(
        f"""
        <div style='font-size:10px;text-transform:uppercase;letter-spacing:0.12em;font-weight:700;opacity:0.6;font-family:monospace;margin-top:14px;'>// Status</div>
        <div style='font-size:11px;color:#67e8f9;line-height:1.7;font-family:monospace;padding:8px 0;'>
          <div>● Model loaded</div>
          <div>● API ready</div>
          <div>● {len(st.session_state.history)} runs logged</div>
        </div>
        <div style='margin-top:18px;font-size:11px;opacity:0.6;line-height:1.5;'>
          Modèle entraîné sur le <i>Water Quality dataset</i>. Plages idéales selon OMS.
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# Main — Hero
# =============================================================================
today = dt.date.today().strftime("%d %b %Y")
st.markdown(
    f"""
    <div class="wf-hero">
        <div>
            <h1>WaterFlow</h1>
            <p>Analyse et prédiction de la potabilité de l'eau par machine learning</p>
        </div>
        <span class="wf-live-tag">Données mises à jour · {today}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Section 01 — World water snapshot
# =============================================================================
st.markdown(
    """
    <div class="wf-section-label">
        <span class="wf-section-num">01</span>
        <h2>État mondial de l'accès à l'eau potable</h2>
        <span class="wf-src">source · WHO/UNICEF JMP 2024</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# 4 KPI cards
c1, c2, c3, c4 = st.columns(4, gap="small")
KPIS = [
    (c1, "danger", "⚠", "2.2", "Mds",   "personnes sans eau potable gérée en toute sécurité",   ("bad", "↑ +30M depuis 2020")),
    (c2, "warn",   "💧","703", "M",     "sans service de base d'eau potable",                    ("good","↓ −80M depuis 2015")),
    (c3, "danger", "☠", "1.4", "M /an", "décès liés à eau, assainissement et hygiène insuffisants", ("bad","×4 enfants < 5 ans")),
    (c4, "good",   "🎯","73",  "%",     "de la population a accès à l'eau gérée en sécurité",    ("good","↑ +5pts vs 2015")),
]
for col, kind, icon, val, unit, label, (dkind, dtxt) in KPIS:
    with col:
        st.markdown(
            f"""
            <div class="wf-kpi">
                <div class="icon {kind}">{icon}</div>
                <div class="value">{val}<span class="unit">{unit}</span></div>
                <div class="label">{label}</div>
                <div class="delta {'bad' if dkind=='bad' else ''}">{dtxt}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# Trend + regions
import plotly.graph_objects as go

left, right = st.columns([1.4, 1], gap="small")

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="ui-monospace, JetBrains Mono, monospace", size=11, color="#07223a"),
    margin=dict(l=8, r=8, t=8, b=8),
    xaxis=dict(showgrid=False, zeroline=False, tickcolor="#d6e6ee", linecolor="#d6e6ee", tickfont=dict(color="#07223a")),
    yaxis=dict(showgrid=True, gridcolor="#eaf3f8", zeroline=False, ticksuffix="%", tickfont=dict(color="#07223a")),
    height=200,
)

with left:
    st.markdown(
        """
        <div class="wf-card" style="padding-bottom:6px;">
            <div class="ttl">Évolution de l'accès à l'eau potable gérée en sécurité</div>
            <div class="sub">// accès (%) vs population mondiale (Mds) · 2000 → 2024</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    years = [2000, 2005, 2010, 2015, 2020, 2022, 2024]
    access = [55, 61, 65, 68, 71, 72, 73]
    population = [6.1, 6.5, 6.9, 7.4, 7.8, 8.0, 8.2]
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=years, y=access, name="Accès (%)",
        mode="lines+markers",
        line=dict(color="#2aa9ff", width=2.5),
        marker=dict(color="#0a4a7a", size=6),
        fill="tozeroy", fillcolor="rgba(42,169,255,0.10)",
        hovertemplate="%{x} : %{y}%<extra>Accès</extra>",
        yaxis="y1",
    ))
    fig_trend.add_trace(go.Scatter(
        x=years, y=population, name="Population (Mds)",
        mode="lines+markers",
        line=dict(color="#f59e0b", width=2, dash="dot"),
        marker=dict(color="#b45309", size=5),
        hovertemplate="%{x} : %{y}Mds<extra>Population</extra>",
        yaxis="y2",
    ))
    fig_trend.update_layout(**{
        **CHART_LAYOUT,
        "height": 220,
        "yaxis": dict(showgrid=True, gridcolor="#eaf3f8", zeroline=False, ticksuffix="%", tickfont=dict(color="#2aa9ff"), title=dict(text="Accès", font=dict(color="#2aa9ff", size=10))),
        "yaxis2": dict(overlaying="y", side="right", showgrid=False, zeroline=False, ticksuffix="Mds", tickfont=dict(color="#f59e0b"), title=dict(text="Population", font=dict(color="#f59e0b", size=10))),
        "legend": dict(orientation="h", y=1.12, x=0, font=dict(size=10, color="#07223a"), bgcolor="rgba(0,0,0,0)"),
    })
    fig_trend.update_xaxes(tickvals=years, tickformat="d", tickfont=dict(color="#07223a"))
    st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

with right:
    st.markdown(
        """
        <div class="wf-card" style="padding-bottom:6px;">
            <div class="ttl">Accès par région</div>
            <div class="sub">// pop. avec eau gérée en sécurité · 2024</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    reg_labels = ["Europe & N. Am.", "Asie de l'Est", "Am. latine", "Asie du Sud", "Océanie", "Afrique sub."]
    reg_values = [96, 88, 78, 67, 54, 31]
    colors = ["#0a4a7a" if v >= 70 else "#2aa9ff" if v >= 50 else "#ef4444" for v in reg_values]
    fig_reg = go.Figure(go.Bar(
        y=reg_labels, x=reg_values, orientation="h",
        marker_color=colors,
        text=[f"{v}%" for v in reg_values],
        textposition="outside",
        hovertemplate="%{y} : %{x}%<extra></extra>",
    ))
    fig_reg.update_layout(**{**CHART_LAYOUT, "font": dict(family="ui-monospace, JetBrains Mono, monospace", size=11, color="#07223a"), "yaxis": dict(showgrid=False, zeroline=False, tickfont=dict(color="#07223a")), "xaxis": dict(showgrid=True, gridcolor="#eaf3f8", zeroline=False, ticksuffix="%", range=[0, 110], tickfont=dict(color="#07223a"))})
    st.plotly_chart(fig_reg, use_container_width=True, config={"displayModeBar": False})

# =============================================================================
# Divider + Section 02 — Predictor
# =============================================================================
st.markdown("<div class='wf-divider'>// section 02 — votre échantillon</div>", unsafe_allow_html=True)

sample_id = f"SAMPLE-{dt.datetime.now().strftime('%Y-%m%d-%H%M')}"
st.markdown(
    f"""
    <div class="wf-pred-header">
        <div class="h-logo">💧</div>
        <div>
            <h2>Cette eau est-elle potable ?</h2>
            <p>Ajuste les 9 paramètres physico-chimiques — le modèle prédit la potabilité.</p>
        </div>
        <span class="sample-id">{sample_id}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Inputs left / result right
in_col, out_col = st.columns([1.4, 1], gap="large")

def render_input(key: str):
    m = META[key]
    val = st.session_state.inputs[key]
    lo, hi = m["ideal"]
    if lo <= val <= hi:
        cls, txt = "ok", f"✓ {lo}–{hi}"
    elif val < lo * 0.6 or val > hi * 1.5:
        cls, txt = "bad", f"✗ hors {lo}–{hi}"
    else:
        cls, txt = "warn", f"⚠ {lo}–{hi}"
    unit = f" {m['unit']}" if m["unit"] else ""
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;align-items:baseline;margin-top:6px;'>"
        f"<span style='font-size:13px;font-weight:600;color:#07223a;'>{m['label']}{unit} <span class='wf-ideal {cls}'>{txt}</span></span>"
        f"<span style='font-family:monospace;font-size:13px;font-weight:700;color:#0a4a7a;'>{val:g}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    new_val = st.slider(
        m["label"],
        min_value=float(m["min"]), max_value=float(m["max"]),
        value=float(val), step=float(m["step"]),
        key=f"slider_{key}",
        label_visibility="collapsed",
    )
    st.session_state.inputs[key] = new_val

with in_col:
    for gkey, (icon, gname, gdesc) in GROUPS.items():
        with st.container(border=True):
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px;font-weight:700;color:#07223a;font-size:12px;"
                f"text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;'>"
                f"{icon} {gname} <span style='font-size:10px;padding:2px 8px;border-radius:999px;"
                f"background:rgba(34,211,238,0.12);color:#0a4a7a;border:1px solid rgba(34,211,238,0.3);font-weight:600;'>{gdesc}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            for k, m in META.items():
                if m["group"] == gkey:
                    render_input(k)

    predict_clicked = st.button(
        "▶  Prédire la potabilité",
        type="primary",
        use_container_width=True,
        key="predict_btn",
    )

# Run prediction
if predict_clicked:
    payload = WaterInput(**st.session_state.inputs)
    result = predict(payload)
    st.session_state.result = result
    st.session_state.source = "local"
    st.session_state.history.append({
        "ts": dt.datetime.now().strftime("%H:%M"),
        "ph": st.session_state.inputs["ph"],
        "organic_carbon": st.session_state.inputs["organic_carbon"],
        "potability": result.potability,
        "probability": result.probability,
    })

with out_col:
    res = st.session_state.result
    src = st.session_state.source

    if res is None:
        st.markdown(
            """
            <div class="wf-result idle">
                <div class="verdict">⏳ En attente</div>
                <div class="sub">// configure ton échantillon puis lance la prédiction</div>
                <div class="prob">--<small>%</small></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        potable = bool(res.potability)
        cls = "safe" if potable else "unsafe"
        verdict = "✅ Potable" if potable else "🚫 Non potable"
        st.markdown(
            f"""
            <div class="wf-result {cls}">
                <div class="verdict">{verdict}</div>
                <div class="sub">// probability · class {res.potability}</div>
                <div class="prob">{res.probability*100:.1f}<small>%</small></div>
                <div class="progress"><div class="fill" style="width:{res.probability*100:.1f}%;"></div></div>
                <div class="grid">
                    <div class="stat"><div class="k">class</div><div class="v">{res.potability}</div></div>
                    <div class="stat"><div class="k">conf</div><div class="v">{res.probability:.3f}</div></div>
                    <div class="stat"><div class="k">thresh</div><div class="v">0.500</div></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        src_color = {"api": "#0369a1", "local": "#b45309", "mock": "#991b1b"}[src]
        src_label = {"api": "via API FastAPI", "local": "via model.pkl local", "mock": "mock (UI demo)"}[src]
        st.markdown(
            f"<div style='margin-top:10px;font-size:11px;font-family:monospace;color:{src_color};font-weight:600;'>↳ {src_label}</div>",
            unsafe_allow_html=True,
        )

    # History
    if st.session_state.history:
        rows = ""
        for h in reversed(st.session_state.history[-5:]):
            emoji = "●" if h["potability"] else "○"
            rows += (
                f"<div style='display:flex;justify-content:space-between;padding:9px 12px;"
                f"border-radius:8px;font-size:12px;background:#f0f7fb;color:#07223a;margin-bottom:6px;'>"
                f"<span>{emoji} {h['ts']} · pH {h['ph']:.1f}</span>"
                f"<span style='font-family:monospace;font-weight:700;color:#0a4a7a;'>{h['probability']:.3f}</span>"
                f"</div>"
            )
        st.markdown(
            f"""
            <div style="margin-top:14px;">
              <div style='display:flex;align-items:center;gap:10px;font-weight:700;color:#07223a;font-size:12px;
                          text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;'>
                  ↻ Historique
                  <span style='font-size:10px;padding:2px 8px;border-radius:999px;background:rgba(34,211,238,0.12);
                               color:#0a4a7a;border:1px solid rgba(34,211,238,0.3);font-weight:600;'>{len(st.session_state.history)} runs</span>
              </div>
              {rows}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Did-you-know callout
    st.markdown(
        """
        <div class="wf-callout" style="margin-top:14px;">
            <div class="ttl">💡 Le savais-tu ?</div>
            <div class="body">
                1 être humain sur 4 dans le monde n'a pas accès à de l'eau potable gérée en sécurité —
                soit près de <b>2 milliards de personnes</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
