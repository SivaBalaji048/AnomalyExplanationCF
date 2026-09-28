"""Streamlit demo application — Swiss Machine frontend.

Responsibilities:
- Swiss Machine visual identity (precise, industrial, research-oriented).
- Integration with the Anomaly Explanation with Counterfactuals ML pipeline.
- Bug A fix: PlausibilityChecker receives only the 5 detector features.
- Bug B fix: Batch anomaly explanation passes only the 5 detector features.
- Bug C fix: PROJECT_ROOT injected into sys.path before any src import.
"""
import streamlit as st
import yaml
import pandas as pd
import json
import time
import sys
from pathlib import Path

# ── Project root on sys.path ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Constants ────────────────────────────────────────────────────────────────
DETECTOR_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FEATURE_UNITS = {
    "Air temperature [K]":     "K",
    "Process temperature [K]": "K",
    "Rotational speed [rpm]":  "RPM",
    "Torque [Nm]":             "Nm",
    "Tool wear [min]":         "min",
}
FEATURE_LABELS = {
    "Air temperature [K]":     "AIR TEMPERATURE",
    "Process temperature [K]": "PROCESS TEMPERATURE",
    "Rotational speed [rpm]":  "ROTATIONAL SPEED",
    "Torque [Nm]":             "TORQUE",
    "Tool wear [min]":         "TOOL WEAR",
}

# ── HTML report ──────────────────────────────────────────────────────────────
def get_html_report(result: dict) -> str:
    body = json.dumps(result, default=str, indent=2)
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Swiss Machine | Explainability Report</title><style>"
        "body{font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;"
        "background:#F7F7F3;color:#111;padding:3rem;line-height:1.55;max-width:900px}"
        "h1{font-size:32px;font-weight:700;letter-spacing:-.02em;margin-bottom:.25rem}"
        "h2{font-size:16px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;"
        "border-bottom:1px solid #111;padding-bottom:.4rem;margin:2rem 0 .75rem}"
        "pre{font-family:'IBM Plex Mono','Courier New',monospace;font-size:12px;"
        "white-space:pre-wrap;word-break:break-all;background:#fff;"
        "border:1px solid #D9D9D4;padding:1rem;margin:.75rem 0}"
        ".notice{font-size:12px;color:#595959;border-top:1px solid #D9D9D4;"
        "margin-top:2rem;padding-top:1rem;line-height:1.55}"
        ".notice ul{padding-left:1.2rem;margin-top:.4rem}"
        ".notice li{margin-bottom:.25rem}"
        "</style></head><body>"
        "<h1>MACHINE STATE / EXPLAINER</h1>"
        "<h2>Structured Result</h2>"
        f"<pre>{body}</pre>"
        "<div class='notice'><strong>Model Limitations</strong><ul>"
        "<li>Model-based, not causal.</li>"
        "<li>Not a certified safety recommendation.</li>"
        "<li>Not a guarantee of repair or failure prevention.</li>"
        "<li>Observed dataset bounds are not physical safety limits.</li>"
        "<li>A valid, feasible counterfactual may still be implausible.</li>"
        "<li>Frozen Isolation Forest baseline with limited recall.</li>"
        "</ul></div></body></html>"
    )

# ── Cached loaders ───────────────────────────────────────────────────────────
@st.cache_resource
def load_features_config():
    with open(PROJECT_ROOT / "config" / "features.yaml", "r") as f:
        return yaml.safe_load(f)

@st.cache_resource
def load_models():
    from src.preprocessing import DataPreprocessor
    from src.detector import AnomalyDetector
    preprocessor = DataPreprocessor.load()
    detector = AnomalyDetector.load()
    return preprocessor, detector

@st.cache_resource
def load_pipeline():
    from src.preprocessing import DataPreprocessor
    from src.detector import AnomalyDetector
    from src.counterfactual import CounterfactualEngine
    from src.feasibility import FeasibilityChecker
    from src.plausibility import PlausibilityChecker
    from src.pipeline import ExplanationPipeline
    from src.data_loader import load_raw_dataset

    preprocessor = DataPreprocessor.load()
    detector = AnomalyDetector.load()
    df_raw = load_raw_dataset()
    # BUG A FIX: only the 5 detector features in exact configured order
    normal_train, _, _ = preprocessor.prepare_datasets(df_raw)
    normal_train_features = normal_train[preprocessor.feature_names].copy()

    engine = CounterfactualEngine()
    feasibility = FeasibilityChecker()
    plausibility = PlausibilityChecker(
        normal_training_data=normal_train_features,
        preprocessor=preprocessor,
    )
    return ExplanationPipeline(
        preprocessor=preprocessor,
        detector=detector,
        engine=engine,
        feasibility_checker=feasibility,
        plausibility_checker=plausibility,
    )

# ── State / navigation ───────────────────────────────────────────────────────
def initialize_state():
    if "current_view" not in st.session_state:
        st.session_state.current_view = "welcome"

def navigate_to(view_name: str):
    if view_name in ("single", "batch", "dashboard", "welcome"):
        st.session_state.pop("explanation_result", None)
        st.session_state.pop("explanation_input_hash", None)
    st.session_state.current_view = view_name
    st.rerun()

# ── CSS ──────────────────────────────────────────────────────────────────────
CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=Inter:wght@400;500;600&family=Oswald:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#F7F7F3;--ink:#111111;--blue:#174BFF;--pale:#DDE5FF;--green:#18A765;--red:#E94B35;--white:#FFFFFF;--gray:#595959;--divider:#D9D9D4;--major-grid:rgba(17,17,17,0.055);--minor-grid:rgba(17,17,17,0.025);--fh:"Oswald",sans-serif;--fb:Inter,Arial,sans-serif;--fm:"IBM Plex Mono","Courier New",monospace}
[data-testid="stAppViewContainer"]>.main, .stApp{background-color:var(--bg)!important;background-image:linear-gradient(var(--major-grid) 1px, transparent 1px),linear-gradient(90deg, var(--major-grid) 1px, transparent 1px),linear-gradient(var(--minor-grid) 1px, transparent 1px),linear-gradient(90deg, var(--minor-grid) 1px, transparent 1px)!important;background-size:80px 80px, 80px 80px, 20px 20px, 20px 20px!important;background-position:center top!important;}
header[data-testid="stHeader"]{background:transparent!important;display:none!important}
#MainMenu{visibility:hidden!important}
footer{display:none!important}
[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"],.stDeployButton{display:none!important}
.block-container{padding-top:2rem!important;padding-bottom:3rem!important;max-width:960px!important;border-left:1px solid var(--major-grid);border-right:1px solid var(--major-grid);background:transparent!important;min-height:100vh;}
body,.stApp{color:var(--ink);font-family:var(--fb)}
h1,h2,h3,h4,h5,h6{font-family:var(--fh)!important;color:var(--ink)!important;font-weight:600!important}
p{font-family:var(--fb);font-size:16px;line-height:1.55;color:var(--ink)}
/* ---- Buttons ---- */
div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button {background-color:var(--ink)!important;color:var(--white)!important;-webkit-text-fill-color:var(--white)!important;border:none!important;border-radius:0!important;font-family:var(--fh)!important;font-weight:700!important;font-size:14px!important;letter-spacing:.04em!important;padding:.6rem 1.5rem!important;text-transform:uppercase;transition:background .15s ease,transform .1s ease}
div[data-testid="stButton"] button:hover, div[data-testid="stDownloadButton"] button:hover {background-color:var(--blue)!important;color:var(--white)!important;-webkit-text-fill-color:var(--white)!important;transform:translateY(-1px)}
div[data-testid="stButton"] button:active, div[data-testid="stDownloadButton"] button:active {transform:translateY(1px)}
div[data-testid="stButton"] button:focus, div[data-testid="stDownloadButton"] button:focus {outline:2px solid var(--blue)!important;outline-offset:2px!important}
/* ---- Number inputs & Forms ---- */
[data-testid="stNumberInput"]>div{border:none!important;background:transparent!important}
[data-testid="stNumberInput"] input{background-color:var(--white)!important;color:var(--ink)!important;-webkit-text-fill-color:var(--ink)!important;border:1px solid var(--ink)!important;border-radius:0!important;font-family:var(--fm)!important;font-size:18px!important;font-weight:500!important;padding:.5rem .75rem!important;caret-color:var(--ink)!important}
[data-testid="stNumberInput"] input:focus{border-color:var(--blue)!important;box-shadow:0 0 0 2px rgba(23,75,255,.12)!important;outline:none!important}
[data-testid="stNumberInput"] button{background:var(--white)!important;border:1px solid var(--divider)!important;color:var(--ink)!important;-webkit-text-fill-color:var(--ink)!important;border-radius:0!important}
[data-testid="stFileUploader"]{background:var(--white)!important;border:1px solid var(--ink)!important;border-radius:0!important;padding:1.5rem!important;text-align:center;}
[data-testid="stFileDropzone"]{background:transparent!important;border:none!important}
[data-testid="stFileUploader"] label{color:var(--ink)!important;font-family:var(--fm)!important;font-size:13px!important;text-transform:uppercase;letter-spacing:.08em}
[data-testid="stRadio"] label{font-family:var(--fm)!important;font-size:12px!important;color:var(--ink)!important;text-transform:uppercase;letter-spacing:.05em}
[data-testid="stSelectbox"]>div>div{background:var(--white)!important;border:1px solid var(--ink)!important;border-radius:0!important;font-family:var(--fm)!important;font-size:13px!important;color:var(--ink)!important}
[data-testid="stDataFrame"]{border:1px solid var(--ink)!important;border-radius:0!important;font-family:var(--fm)!important;font-size:12px!important}
div[data-testid="stFormSubmitButton"] button {background-color:var(--ink)!important;color:var(--white)!important;-webkit-text-fill-color:var(--white)!important;border:none!important;border-radius:0!important;font-family:var(--fh)!important;font-weight:700!important;font-size:14px!important;letter-spacing:.06em!important;padding:.75rem 2rem!important;text-transform:uppercase}
div[data-testid="stFormSubmitButton"] button:hover {background-color:var(--blue)!important;color:var(--white)!important;-webkit-text-fill-color:var(--white)!important}
[data-testid="stNumberInput"]>label,[data-testid="stSelectbox"]>label,[data-testid="stRadio"]>label,[data-testid="stFileUploader"]>label{font-family:var(--fm)!important;font-size:11px!important;font-weight:500!important;text-transform:uppercase!important;letter-spacing:.07em!important;color:var(--gray)!important;margin-bottom:.25rem!important}
[data-testid="stExpander"]{border:1px solid var(--ink)!important;border-radius:0!important;background:var(--white)!important}
[data-testid="stExpander"] summary{font-family:var(--fm)!important;font-size:12px!important;text-transform:uppercase;letter-spacing:.05em}
[data-testid="stAlert"]{border-radius:0!important;font-family:var(--fb)!important}
/* ---- Layout Spacing ---- */
[data-testid="column"]{padding:0!important}
/* ---- SM Utilities ---- */
.sm-rule{border:none;border-top:1px solid var(--ink);margin:2rem 0}
.sm-divider{border:none;border-top:1px solid var(--divider);margin:1.5rem 0}
.sm-label{font-family:var(--fm);font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:.07em;color:var(--gray);margin-bottom:.2rem}
.sm-value{font-family:var(--fm);font-size:28px;font-weight:500;color:var(--ink);line-height:1.1}
.sm-body{font-family:var(--fb);font-size:15px;line-height:1.55;color:var(--gray)}
.sm-tag-anom{display:inline-block;font-family:var(--fm);font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--red);border:1px solid var(--red);padding:.2rem .5rem;border-radius:0}
.sm-tag-norm{display:inline-block;font-family:var(--fm);font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--green);border:1px solid var(--green);padding:.2rem .5rem;border-radius:0}
.sm-tag-blue{display:inline-block;font-family:var(--fm);font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--blue);border:1px solid var(--blue);padding:.2rem .5rem;border-radius:0}
/* ---- Instrument Cells & Technical Layouts ---- */
.sm-field-label{font-family:var(--fm);font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--gray);margin-bottom:.35rem;display:flex;justify-content:space-between;align-items:baseline}
.sm-field-unit{font-family:var(--fm);font-size:10px;color:var(--divider);letter-spacing:.06em}
.sm-range-hint{font-family:var(--fm);font-size:10px;color:var(--gray);letter-spacing:.04em;margin-top:.2rem}
.sm-section-heading{font-family:var(--fm);font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--ink);border-top:1px solid var(--ink);padding-top:.75rem;margin-top:2.5rem;margin-bottom:1.25rem}
.sm-state-table{width:100%;border-collapse:collapse;font-family:var(--fm);font-size:13px;margin:1rem 0; background: var(--white); border: 1px solid var(--ink);}
.sm-state-table td{padding:.6rem 1rem;border-bottom:1px solid var(--divider);vertical-align:baseline}
.sm-state-table td:first-child{font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:var(--gray);width:55%}
.sm-state-table td:last-child{font-size:16px;font-weight:500;color:var(--ink);text-align:right}
.sm-status-row{display:grid; grid-template-columns: repeat(3, 1fr); border: 1px solid var(--ink); margin: 1.5rem 0;}
.sm-status-cell{display:flex;flex-direction:column;gap:.25rem;padding: 1.25rem; border-right: 1px solid var(--ink); background: var(--white);}
.sm-status-cell:last-child{border-right: none;}
.sm-status-ok{font-family:var(--fm);font-size:14px;font-weight:700;color:var(--green);text-transform:uppercase}
.sm-status-fail{font-family:var(--fm);font-size:14px;font-weight:700;color:var(--red);text-transform:uppercase}
.sm-cf-block{border-top:1px solid var(--ink); border-bottom:1px solid var(--ink); padding:1.5rem 0; margin-bottom: 2rem;}
.sm-cf-feat{font-family:var(--fm);font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--gray);margin-bottom:.5rem}
.sm-cf-vals{display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap}
.sm-cf-orig{font-family:var(--fm);font-size:32px;font-weight:500;color:var(--gray);text-decoration:line-through;text-decoration-color:var(--red)}
.sm-cf-arrow{font-family:var(--fm);font-size:20px;color:var(--blue)}
.sm-cf-new{font-family:var(--fm);font-size:36px;font-weight:700;color:var(--ink)}
.sm-cf-delta{font-family:var(--fm);font-size:13px;font-weight:500;color:var(--blue);margin-top:.25rem}
.sm-metrics-row{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:1rem;margin:1.5rem 0}
.sm-metric{display:flex;flex-direction:column;gap:.2rem;background:var(--white);border:1px solid var(--ink);padding:1rem;}
.sm-metric-label{font-family:var(--fm);font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:var(--gray)}
.sm-metric-value{font-family:var(--fm);font-size:20px;font-weight:500;color:var(--ink)}
@keyframes sm-appear{from{opacity:0;transform:scale(.97)}to{opacity:1;transform:scale(1)}}
.sm-welcome-mark{animation:sm-appear .9s cubic-bezier(.16,1,.3,1) forwards;margin-bottom:2.5rem}
.sm-hero{font-family:var(--fh);font-size:52px;font-weight:700;line-height:1.02;letter-spacing:-.025em;color:var(--ink);margin-bottom:1.5rem}
.sm-hero-sub{font-family:var(--fb);font-size:16px;color:var(--gray);line-height:1.5;max-width:400px;margin-bottom:3rem}
.sm-workflow-title{font-family:var(--fh);font-size:24px;font-weight:700;color:var(--ink);margin-bottom:.5rem}
.sm-workflow-desc{font-family:var(--fb);font-size:14px;color:var(--gray);line-height:1.5}
.sm-workflow-num{font-family:var(--fm);font-size:11px;color:var(--gray);letter-spacing:.04em}
.sm-loading{position:fixed;inset:0;z-index:999999;display:flex;flex-direction:column;align-items:center;justify-content:center;background-color:var(--bg)}
.sm-loading__container{position:relative;width:120px;height:160px;margin-bottom:1.5rem}
.sm-loading__stage{position:absolute;top:0;left:0;width:120px;display:flex;flex-direction:column;align-items:center;opacity:0}
.sm-loading__svg{width:120px;height:120px;margin-bottom:1.5rem}
.sm-loading__svg *{transform-origin:60px 60px}
.sm-loading__label{font-family:var(--fm);font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.15em;color:var(--ink);margin-bottom:.75rem}
.sm-loading__rule{width:100%;border:none;border-top:1px solid var(--ink);margin:0}
.sm-loading__detect{animation:sm-loading-vis-1 7.8s infinite}
.sm-loading__analyze{animation:sm-loading-vis-2 7.8s infinite}
.sm-loading__explain{animation:sm-loading-vis-3 7.8s infinite}
@keyframes sm-loading-vis-1{0%,33.33%{opacity:1}33.34%,100%{opacity:0}}
@keyframes sm-loading-vis-2{0%,33.33%{opacity:0}33.34%,66.66%{opacity:1}66.67%,100%{opacity:0}}
@keyframes sm-loading-vis-3{0%,66.66%{opacity:0}66.67%,100%{opacity:1}}
.sm-loading__anim-det-draw{animation:sm-loading-det-draw 7.8s infinite}
@keyframes sm-loading-det-draw{0%{stroke-dasharray:400;stroke-dashoffset:400;opacity:0}5%{opacity:1}20%{stroke-dashoffset:0}33.33%{stroke-dasharray:400;stroke-dashoffset:0;opacity:1}33.34%,100%{opacity:0}}
.sm-loading__anim-det-scale{animation:sm-loading-det-scale 7.8s infinite}
@keyframes sm-loading-det-scale{0%{transform:scale(0.5);opacity:0}15%{transform:scale(1.1);opacity:1}25%,33.33%{transform:scale(1);opacity:1}33.34%,100%{opacity:0}}
.sm-loading__anim-ana-draw{animation:sm-loading-ana-draw 7.8s infinite}
@keyframes sm-loading-ana-draw{0%,33.33%{stroke-dasharray:300;stroke-dashoffset:300;opacity:0}38%{opacity:1}53%{stroke-dashoffset:0}66.66%{stroke-dasharray:300;stroke-dashoffset:0;opacity:1}66.67%,100%{opacity:0}}
.sm-loading__anim-ana-wedge{animation:sm-loading-ana-wedge 7.8s infinite}
@keyframes sm-loading-ana-wedge{0%,33.33%{transform:translate(-10px,-10px);opacity:0}45%{transform:translate(-10px,-10px);opacity:0}55%,66.66%{transform:translate(0,0);opacity:1}66.67%,100%{opacity:0}}
.sm-loading__anim-exp-head{animation:sm-loading-exp-head 7.8s infinite}
@keyframes sm-loading-exp-head{0%,66.66%{stroke-dasharray:300;stroke-dashoffset:300;opacity:0}71%{opacity:1}86%,100%{stroke-dasharray:300;stroke-dashoffset:0;opacity:1}}
.sm-loading__anim-exp-comm{animation:sm-loading-exp-comm 7.8s infinite}
@keyframes sm-loading-exp-comm{0%,66.66%{opacity:0;transform:translateX(-10px)}80%{opacity:0;transform:translateX(-10px)}90%,100%{opacity:1;transform:translateX(0)}}
.sm-summary-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:1rem;margin:1.5rem 0;border-top:1px solid var(--divider);padding-top:1.5rem}
.sm-summary-item{display:flex;flex-direction:column;gap:.2rem;background:var(--white);border:1px solid var(--ink);padding:.75rem;}
.sm-interpretation{background:var(--white);border:1px solid var(--ink);border-left:4px solid var(--blue);padding:1rem 1.25rem;font-family:var(--fb);font-size:15px;line-height:1.6;color:var(--ink);margin:1.5rem 0}
.sm-notice{font-family:var(--fb);font-size:13px;color:var(--gray);border-top:1px solid var(--divider);padding-top:1.25rem;margin-top:2rem;line-height:1.55}
.sm-notice ul{padding-left:1.1rem;margin-top:.5rem}
.sm-notice li{margin-bottom:.3rem}
</style>
"""

def load_css():
    st.markdown(CSS, unsafe_allow_html=True)

# ── Logo ─────────────────────────────────────────────────────────────────────
def logo_svg(size: int = 32) -> str:
    s = size
    return (
        f'<svg width="{s}" height="{s}" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="50" cy="50" r="44" fill="none" stroke="#111111" stroke-width="5"/>'
        '<path d="M50 6 L50 50 L94 50" fill="none" stroke="#111111" stroke-width="5"/>'
        '<path d="M50 50 L19 19" fill="none" stroke="#174BFF" stroke-width="5"/>'
        '<circle cx="50" cy="50" r="10" fill="#111111"/></svg>'
    )

# ── Header ───────────────────────────────────────────────────────────────────
def render_header():
    view = st.session_state.get("current_view", "dashboard")

    st.markdown(
        f'<div style="display:flex; justify-content:space-between; align-items:flex-end; border-bottom:1px solid var(--ink); padding-bottom:.75rem; margin-bottom:3rem;">'
        f'<div style="font-family:var(--fm); font-size:11px; font-weight:500; text-transform:uppercase; letter-spacing:.08em; display:flex; align-items:center; gap:10px;">'
        f'{logo_svg(20)} MACHINE STATE / EXPLAINER</div>'
        f'<div style="font-family:var(--fm); font-size:10px; color:var(--gray); letter-spacing:.1em;">SYS_ID 88.02</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    
    st.markdown("""
    <style>
    /* Radio as Navigation Links */
    div[data-testid="stRadio"] {
        margin-bottom: 2.5rem;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 2rem !important;
    }
    div[data-testid="stRadio"] label {
        cursor: pointer;
    }
    div[data-testid="stRadio"] div[data-baseweb="radio"] {
        background: transparent !important;
    }
    /* Hide the actual radio circle */
    div[data-testid="stRadio"] div[data-baseweb="radio"] > div:first-child {
        display: none !important;
    }
    /* Style the text */
    div[data-testid="stRadio"] label p {
        font-family: var(--fm) !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: .07em !important;
        color: var(--gray) !important;
        padding: 0 !important;
        margin: 0 !important;
        padding-bottom: 4px !important;
        border-bottom: 2px solid transparent !important;
        transition: color 0.15s ease, border-color 0.15s ease;
    }
    div[data-testid="stRadio"] label:hover p {
        color: var(--ink) !important;
    }
    /* Active State */
    div[data-testid="stRadio"] label input:checked + div p {
        color: var(--blue) !important;
        border-bottom: 2px solid var(--blue) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Map current_view to radio options
    options = ["ANALYZE", "BATCH", "ANALYTICS"]
    view_map = {"dashboard": "ANALYZE", "batch": "BATCH", "analytics": "ANALYTICS"}
    rev_map = {"ANALYZE": "dashboard", "BATCH": "batch", "ANALYTICS": "analytics"}
    
    current_sel = view_map.get(view, "ANALYZE")
    
    def on_nav_change():
        sel = st.session_state.nav_radio
        st.session_state["current_view"] = rev_map[sel]
        
    st.radio("Navigation", options, index=options.index(current_sel), 
             horizontal=True, label_visibility="collapsed", 
             key="nav_radio", on_change=on_nav_change)
def section_heading(label: str):
    st.markdown(f'<div class="sm-section-heading">{label}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# WELCOME
# ═══════════════════════════════════════════════════════════════════
def view_welcome():
    st.markdown('<div style="height: 4rem;"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 11])
    with c1:
        st.markdown(
            '<div style="font-family:var(--fm); font-size:10px; color:var(--gray); writing-mode:vertical-rl; transform:scale(-1); letter-spacing:.1em;">X 000 / Y 120</div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            '<div style="border-left: 1px solid var(--ink); padding-left: 3rem; margin-bottom: 4rem;">'
            f'<div class="sm-welcome-mark">{logo_svg(56)}</div>'
            '<div class="sm-hero">Automated Anomaly<br>Explanation<br>With Counter Factuals</div>'
            '<div class="sm-hero-sub">Explainable machine-state analysis<br>using frozen anomaly detection.</div>'
            '</div>',
            unsafe_allow_html=True
        )
        if st.button("ENTER \u2192", key="btn_enter"):
            navigate_to("dashboard")
# ═══════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════
def view_dashboard():
    render_header()
    st.markdown(
        '<h1 style="font-size:48px;line-height:1.02;letter-spacing:-.025em;margin-bottom:.25rem;">'
        'HOW DO YOU WANT<br>TO ANALYZE?</h1>'
        '<p class="sm-body" style="margin-bottom:2.5rem;">Select an analysis mode to begin.</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="border-top:1px solid var(--ink); padding: 1.5rem 0;">', unsafe_allow_html=True)
    col_num, col_body, col_action = st.columns([1, 7, 2])
    with col_num: st.markdown('<div class="sm-workflow-num">01</div>', unsafe_allow_html=True)
    with col_body: st.markdown('<div class="sm-workflow-title">SINGLE INPUT</div><div class="sm-workflow-desc">Enter one machine state manually and generate a model-based counterfactual explanation.</div>', unsafe_allow_html=True)
    with col_action:
        if st.button("ENTER \u2192", key="btn_single"): navigate_to("single")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="border-top:1px solid var(--divider); border-bottom:1px solid var(--ink); padding: 1.5rem 0; margin-bottom: 2rem;">', unsafe_allow_html=True)
    col_num2, col_body2, col_action2 = st.columns([1, 7, 2])
    with col_num2: st.markdown('<div class="sm-workflow-num">02</div>', unsafe_allow_html=True)
    with col_body2: st.markdown('<div class="sm-workflow-title">BATCH INPUT</div><div class="sm-workflow-desc">Upload a CSV dataset, screen all records for anomalies, then select one for a full explanation.</div>', unsafe_allow_html=True)
    with col_action2:
        if st.button("ENTER \u2192", key="btn_batch"): navigate_to("batch")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<p class="sm-body" style="font-size:13px;">Single analysis produces one model-based explanation. Batch screening identifies anomalous records; explanations are generated only for a selected record.</p>', unsafe_allow_html=True)
# ═══════════════════════════════════════════════════════════════════
# SINGLE INPUT
# ═══════════════════════════════════════════════════════════════════
def view_single():
    render_header()
    st.markdown(
        '<h1 style="font-size:44px;line-height:1.02;letter-spacing:-.025em;margin-bottom:.25rem;">ANALYZE / SINGLE</h1>'
        '<p class="sm-body" style="margin-bottom:2.5rem;">Enter machine state values to generate a model-based counterfactual explanation.</p>',
        unsafe_allow_html=True,
    )
    config = load_features_config()
    feats = config["features"]

    def mid(name):
        return round((feats[name]["min"] + feats[name]["max"]) / 2.0, 4)

    section_heading("01 / THERMAL STATE")
    st.markdown(
        '<p class="sm-body" style="font-size:13px;margin-bottom:1rem;">'
        'Temperature features are <em>immutable</em> \u2014 the counterfactual engine does not suggest changes to these values.</p>',
        unsafe_allow_html=True,
    )

    with st.form("single_input_form"):
        ca, cb = st.columns(2, gap="large")
        with ca:
            st.markdown('<div class="sm-field-label">AIR TEMPERATURE <span class="sm-field-unit">K</span></div>', unsafe_allow_html=True)
            air_temp = st.number_input("Air temperature [K]",
                min_value=float(feats["Air temperature [K]"]["min"]),
                max_value=float(feats["Air temperature [K]"]["max"]),
                value=mid("Air temperature [K]"), step=0.1, label_visibility="collapsed")
            st.markdown(
                f'<div class="sm-range-hint">OBSERVED RANGE {feats["Air temperature [K]"]["min"]} \u2014 {feats["Air temperature [K]"]["max"]} K</div>',
                unsafe_allow_html=True)
        with cb:
            st.markdown('<div class="sm-field-label">PROCESS TEMPERATURE <span class="sm-field-unit">K</span></div>', unsafe_allow_html=True)
            process_temp = st.number_input("Process temperature [K]",
                min_value=float(feats["Process temperature [K]"]["min"]),
                max_value=float(feats["Process temperature [K]"]["max"]),
                value=mid("Process temperature [K]"), step=0.1, label_visibility="collapsed")
            st.markdown(
                f'<div class="sm-range-hint">OBSERVED RANGE {feats["Process temperature [K]"]["min"]} \u2014 {feats["Process temperature [K]"]["max"]} K</div>',
                unsafe_allow_html=True)

        section_heading("02 / OPERATING STATE")
        st.markdown(
            '<p class="sm-body" style="font-size:13px;margin-bottom:1rem;">'
            'These features are <em>mutable</em> \u2014 the counterfactual engine may suggest changes to Rotational Speed and Torque.</p>',
            unsafe_allow_html=True,
        )
        cc, cd = st.columns(2, gap="large")
        with cc:
            st.markdown('<div class="sm-field-label">ROTATIONAL SPEED <span class="sm-field-unit">RPM</span></div>', unsafe_allow_html=True)
            rot_speed = st.number_input("Rotational speed [rpm]",
                min_value=float(feats["Rotational speed [rpm]"]["min"]),
                max_value=float(feats["Rotational speed [rpm]"]["max"]),
                value=mid("Rotational speed [rpm]"), step=10.0, label_visibility="collapsed")
            st.markdown(
                f'<div class="sm-range-hint">OBSERVED RANGE {feats["Rotational speed [rpm]"]["min"]} \u2014 {feats["Rotational speed [rpm]"]["max"]} RPM</div>',
                unsafe_allow_html=True)
        with cd:
            st.markdown('<div class="sm-field-label">TORQUE <span class="sm-field-unit">Nm</span></div>', unsafe_allow_html=True)
            torque = st.number_input("Torque [Nm]",
                min_value=float(feats["Torque [Nm]"]["min"]),
                max_value=float(feats["Torque [Nm]"]["max"]),
                value=mid("Torque [Nm]"), step=0.1, label_visibility="collapsed")
            st.markdown(
                f'<div class="sm-range-hint">OBSERVED RANGE {feats["Torque [Nm]"]["min"]} \u2014 {feats["Torque [Nm]"]["max"]} Nm</div>',
                unsafe_allow_html=True)

        section_heading("03 / WEAR STATE")
        st.markdown(
            '<p class="sm-body" style="font-size:13px;margin-bottom:1rem;">'
            'Tool wear is cumulative and <em>immutable</em> in the current counterfactual policy.</p>',
            unsafe_allow_html=True,
        )
        ce, _ = st.columns(2, gap="large")
        with ce:
            st.markdown('<div class="sm-field-label">TOOL WEAR <span class="sm-field-unit">min</span></div>', unsafe_allow_html=True)
            tool_wear = st.number_input("Tool wear [min]",
                min_value=float(feats["Tool wear [min]"]["min"]),
                max_value=float(feats["Tool wear [min]"]["max"]),
                value=mid("Tool wear [min]"), step=1.0, label_visibility="collapsed")
            st.markdown(
                f'<div class="sm-range-hint">OBSERVED RANGE {feats["Tool wear [min]"]["min"]} \u2014 {feats["Tool wear [min]"]["max"]} min</div>',
                unsafe_allow_html=True)

        st.markdown('<hr class="sm-divider">', unsafe_allow_html=True)
        st.markdown(
            '<p class="sm-body" style="font-size:12px;color:var(--gray);">'
            'OBSERVED DATASET RANGES ARE USED FOR INPUT VALIDATION. THEY ARE NOT CERTIFIED PHYSICAL SAFETY LIMITS.</p>',
            unsafe_allow_html=True,
        )
        submitted = st.form_submit_button("GENERATE \u2192")
        if submitted:
            st.session_state.pop("explanation_result", None)
            st.session_state.pop("explanation_input_hash", None)
            st.session_state.single_input_df = pd.DataFrame([{
                "Air temperature [K]":     air_temp,
                "Process temperature [K]": process_temp,
                "Rotational speed [rpm]":  rot_speed,
                "Torque [Nm]":             torque,
                "Tool wear [min]":         tool_wear,
            }])
            navigate_to("explanation")

# ═══════════════════════════════════════════════════════════════════
# BATCH
# ═══════════════════════════════════════════════════════════════════
def view_batch():
    render_header()
    st.markdown(
        '<h1 style="font-size:44px;line-height:1.02;letter-spacing:-.025em;margin-bottom:.25rem;">ANALYZE / BATCH</h1>'
        '<p class="sm-body" style="margin-bottom:2.5rem;">Upload a CSV dataset. The screening engine will classify each valid record.</p>',
        unsafe_allow_html=True,
    )
    config = load_features_config()
    required_features = config["feature_sets"]["detector_inputs"]
    features_meta = config["features"]

    section_heading("UPLOAD DATASET")
    st.markdown('<p class="sm-body" style="font-size:13px;margin-bottom:.75rem;">CSV must contain the five detector feature columns.</p>', unsafe_allow_html=True)
    st.markdown('<div style="background:var(--white); padding:1.5rem; border:1px solid var(--ink); margin-bottom:1rem; text-align:center;">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("CSV DATASET", type=["csv"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        if st.session_state.get("_batch_fname") != uploaded_file.name:
            st.session_state["_batch_fname"] = uploaded_file.name
            st.session_state.pop("screening_df", None)
            st.session_state.pop("screen_results", None)
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            return

        missing_cols = [c for c in required_features if c not in df.columns]
        if missing_cols:
            st.error(f"SCHEMA ERROR \u2014 Missing required columns: {missing_cols}")
            return

        valid_rows, invalid_reasons = [], []
        for idx, row in df.iterrows():
            reasons = []
            for f in required_features:
                val = row[f]
                if pd.isna(val):
                    reasons.append(f"{f}: missing")
                    continue
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    reasons.append(f"{f}: not numeric")
                    continue
                if val < features_meta[f]["min"] or val > features_meta[f]["max"]:
                    reasons.append(f"{f}: out of range")
            if reasons:
                invalid_reasons.append({"row_index": idx, "reason": " | ".join(reasons)})
            else:
                valid_rows.append(idx)

        st.markdown(
            f'<div class="sm-summary-grid">'
            f'<div class="sm-summary-item"><div class="sm-metric-label">FILE</div>'
            f'<div style="font-family:var(--fm);font-size:13px;color:var(--ink);">{uploaded_file.name}</div></div>'
            f'<div class="sm-summary-item"><div class="sm-metric-label">TOTAL ROWS</div>'
            f'<div class="sm-metric-value">{len(df):,}</div></div>'
            f'<div class="sm-summary-item"><div class="sm-metric-label">VALID</div>'
            f'<div class="sm-metric-value" style="color:var(--green);">{len(valid_rows):,}</div></div>'
            f'<div class="sm-summary-item"><div class="sm-metric-label">INVALID</div>'
            f'<div class="sm-metric-value" style="color:var(--red);">{len(invalid_reasons):,}</div></div>'
            f'<div class="sm-summary-item"><div class="sm-metric-label">STATUS</div>'
            f'<div style="font-family:var(--fm);font-size:13px;color:var(--green);font-weight:700;">{"READY" if valid_rows else "NO VALID ROWS"}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if invalid_reasons:
            with st.expander(f"VIEW {len(invalid_reasons)} INVALID ROWS"):
                st.dataframe(pd.DataFrame(invalid_reasons), use_container_width=True)

        if valid_rows:
            if st.button("RUN SCREENING \u2192", key="btn_run_screening"):
                st.session_state.screening_df = df.loc[valid_rows].copy()
                st.session_state.pop("screen_results", None)
                st.rerun()

    if "screening_df" not in st.session_state:
        return

    section_heading("SCREENING RESULTS")
    df_screen = st.session_state.screening_df
    preprocessor, detector = load_models()

    if "screen_results" not in st.session_state:
        with st.spinner("Running anomaly screening\u2026"):
            scaled = preprocessor.transform(df_screen)
            scores = detector.predict_anomaly_score(scaled)
            is_anom = detector.is_anomaly(scaled)
            results_df = df_screen[required_features].copy()
            results_df["DETECTOR_DECISION"] = is_anom.map({True: "ANOMALY DETECTED", False: "NORMAL"})
            results_df["ANOMALY_SCORE"] = scores.round(6)
            st.session_state.screen_results = results_df

    results_df = st.session_state.screen_results
    total = len(results_df)
    n_anom = (results_df["DETECTOR_DECISION"] == "ANOMALY DETECTED").sum()
    n_norm = total - n_anom

    st.markdown(
        f'<div style="margin-bottom:1.5rem;">'
        f'<span style="font-family:var(--fm);font-size:28px;font-weight:700;">{total:,}</span>'
        f'<span class="sm-label" style="margin-left:.5rem;">RECORDS ANALYZED</span>'
        f'</div>'
        f'<div class="sm-status-row">'
        f'<div class="sm-status-cell"><div class="sm-label">NORMAL</div>'
        f'<div class="sm-metric-value" style="color:var(--green);">{n_norm:,}</div></div>'
        f'<div class="sm-status-cell"><div class="sm-label">ANOMALIES</div>'
        f'<div class="sm-metric-value" style="color:var(--red);">{n_anom:,}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    filt = st.radio("Filter", ["ALL", "ANOMALIES", "NORMAL"], horizontal=True, label_visibility="collapsed")
    if filt == "ANOMALIES":
        display_df = results_df[results_df["DETECTOR_DECISION"] == "ANOMALY DETECTED"]
    elif filt == "NORMAL":
        display_df = results_df[results_df["DETECTOR_DECISION"] == "NORMAL"]
    else:
        display_df = results_df

    st.dataframe(display_df, use_container_width=True)
    st.download_button("DOWNLOAD RESULTS (CSV)",
        data=results_df.to_csv(index=True).encode("utf-8"),
        file_name="screening_results.csv", mime="text/csv")

    if n_anom == 0:
        return

    st.markdown('<hr class="sm-rule">', unsafe_allow_html=True)
    section_heading("EXPLAIN ANOMALY")
    anomaly_indices = results_df[results_df["DETECTOR_DECISION"] == "ANOMALY DETECTED"].index.tolist()
    selected_idx = st.selectbox("ORIGINAL ROW INDEX", options=anomaly_indices)

    if selected_idx is not None:
        sel_row = results_df.loc[selected_idx]
        st.markdown(
            f'<div style="margin:1rem 0 1.5rem;">'
            f'<div class="sm-label">SELECTED RECORD</div>'
            f'<div style="font-family:var(--fm);font-size:14px;margin-top:.5rem;">ROW {selected_idx}</div>'
            f'<div class="sm-tag-anom" style="margin-top:.5rem;">ANOMALY DETECTED</div>'
            f'<div style="font-family:var(--fm);font-size:13px;color:var(--gray);margin-top:.5rem;">'
            f'ANOMALY SCORE &nbsp;{sel_row["ANOMALY_SCORE"]}</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("GENERATE EXPLANATION \u2192", key="btn_gen_explain"):
            # BUG B FIX: only the 5 detector features in exact order
            record = results_df.loc[[selected_idx], DETECTOR_FEATURES].copy()
            st.session_state.pop("explanation_result", None)
            st.session_state.pop("explanation_input_hash", None)
            st.session_state.single_input_df = record
            navigate_to("explanation")

# ═══════════════════════════════════════════════════════════════════
# EXPLANATION (includes loading)
# ═══════════════════════════════════════════════════════════════════
def view_explanation():
    render_header()

    if "single_input_df" not in st.session_state:
        st.error("No input state. Please return to Analyze and enter a machine state.")
        if st.button("\u2190 BACK TO ANALYZE", key="btn_back_nostate"):
            navigate_to("dashboard")
        return

    df_input = st.session_state.single_input_df.copy()
    input_hash = str(df_input.values.tolist())

    if (
        "explanation_result" not in st.session_state
        or st.session_state.get("explanation_input_hash") != input_hash
    ):
        loader_ph = st.empty()
        with loader_ph.container():
            st.markdown('<div style="height:3rem;"></div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1, 1, 1])
            with c2:
                loader_html = """
                <div class="sm-loading">
                    <div class="sm-loading__container">
                        <!-- DETECT -->
                        <div class="sm-loading__stage sm-loading__detect">
                            <svg class="sm-loading__svg" viewBox="0 0 120 120">
                                <path class="sm-loading__anim-det-draw" d="M 60 10 L 60 40 M 60 80 L 60 110 M 10 60 L 40 60 M 80 60 L 110 60" stroke="#111" stroke-width="1.5" stroke-linecap="square" fill="none" />
                                <path d="M 10 20 L 10 10 L 20 10 M 110 20 L 110 10 L 100 10 M 10 100 L 10 110 L 20 110 M 110 100 L 110 110 L 100 110" stroke="#111" stroke-width="1.5" fill="none" />
                                <circle class="sm-loading__anim-det-scale" cx="60" cy="60" r="30" stroke="#111" stroke-width="1.5" fill="none" />
                                <circle class="sm-loading__anim-det-scale" cx="60" cy="60" r="24" stroke="#111" stroke-width="2" fill="none" stroke-dasharray="2 4" />
                                <circle class="sm-loading__anim-det-scale" cx="60" cy="60" r="8" fill="#111" />
                                <rect x="48" y="58" width="4" height="4" fill="#174BFF" />
                                <rect x="58" y="48" width="4" height="4" fill="#FF00FF" />
                                <rect x="68" y="58" width="4" height="4" fill="#FFFF00" />
                            </svg>
                            <div class="sm-loading__label">DETECT</div>
                            <hr class="sm-loading__rule">
                        </div>
                        <!-- ANALYZE -->
                        <div class="sm-loading__stage sm-loading__analyze">
                            <svg class="sm-loading__svg" viewBox="0 0 120 120">
                                <path class="sm-loading__anim-ana-draw" d="M 60 20 A 40 40 0 1 1 20 60" fill="none" stroke="#111" stroke-width="24" stroke-linecap="square" />
                                <path class="sm-loading__anim-ana-wedge" d="M 60 60 L 24 24 A 40 40 0 0 1 60 20 Z" fill="#111" />
                            </svg>
                            <div class="sm-loading__label">ANALYZE</div>
                            <hr class="sm-loading__rule">
                        </div>
                        <!-- EXPLAIN -->
                        <div class="sm-loading__stage sm-loading__explain">
                            <svg class="sm-loading__svg" viewBox="0 0 120 120">
                                <path class="sm-loading__anim-exp-head" d="M 40 90 L 40 50 C 40 35 55 25 70 30 C 85 35 85 55 70 70 L 60 70 L 60 90 Z" fill="none" stroke="#111" stroke-width="10" stroke-linejoin="miter" />
                                <circle cx="70" cy="50" r="4" fill="#174BFF" />
                                <rect x="68" y="60" width="4" height="12" fill="#174BFF" />
                                <path class="sm-loading__anim-exp-comm" d="M 90 40 L 110 30 M 95 55 L 115 55 M 90 70 L 110 80" stroke="#174BFF" stroke-width="2" stroke-linecap="square" />
                                <line x1="20" y1="90" x2="100" y2="90" stroke="#111" stroke-width="2" />
                            </svg>
                            <div class="sm-loading__label">EXPLAIN</div>
                            <hr class="sm-loading__rule">
                        </div>
                    </div>
                </div>
                """
                st.markdown(loader_html, unsafe_allow_html=True)
            st.markdown('<div style="height:3rem;"></div>', unsafe_allow_html=True)
        try:
            pipeline = load_pipeline()
            result = pipeline.analyze(df_input)
            st.session_state.explanation_result = result
            st.session_state.explanation_input_hash = input_hash
        except Exception as e:
            loader_ph.empty()
            st.error(f"PIPELINE ERROR \u2014 {e}")
            if st.button("\u2190 BACK TO ANALYZE", key="btn_back_err"):
                navigate_to("dashboard")
            return
        loader_ph.empty()

    result = st.session_state.explanation_result
    config = load_features_config()
    mutable_features = config["counterfactual_policy"]["mutable_features"]

    is_normal = (
        not result.get("found", False)
        and result.get("reason") == "original_state_is_already_normal"
    )

    st.markdown(
        '<h1 style="font-size:44px;line-height:1.02;letter-spacing:-.025em;margin-bottom:.5rem;">EXPLANATION / 001</h1>',
        unsafe_allow_html=True,
    )

    if is_normal:
        st.markdown('<div class="sm-tag-norm" style="font-size:14px;padding:.4rem .8rem;">NORMAL</div>', unsafe_allow_html=True)
        st.markdown('<p class="sm-body" style="margin-top:1rem;">The frozen anomaly detector classified this machine state as normal. No counterfactual explanation is required.</p>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sm-tag-anom" style="font-size:14px;padding:.4rem .8rem;">ANOMALY DETECTED</div>', unsafe_allow_html=True)
        st.markdown('<p class="sm-body" style="margin-top:1rem;">The machine state lies outside the detector\'s learned normal region.</p>', unsafe_allow_html=True)

    # 1. CURRENT STATE
    section_heading("CURRENT STATE")
    rows_html = "".join([
        f'<tr><td>{FEATURE_LABELS.get(col, col)}</td>'
        f'<td>{df_input[col].iloc[0]:g} {FEATURE_UNITS.get(col,"")}</td></tr>'
        for col in DETECTOR_FEATURES if col in df_input.columns
    ])
    st.markdown(f'<table class="sm-state-table">{rows_html}</table>', unsafe_allow_html=True)

    if is_normal:
        st.markdown(
            f'<div class="sm-metrics-row" style="margin-top:1.5rem;">'
            f'<div class="sm-metric"><div class="sm-metric-label">DETECTOR SCORE</div>'
            f'<div class="sm-metric-value">{result.get("original_score", "N/A")}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("ANALYZE ANOTHER \u2192", key="btn_analyze_another"):
            navigate_to("single")
        return

    # 2. MODEL COUNTERFACTUAL
    section_heading("MODEL COUNTERFACTUAL")

    if result.get("found"):
        cf_state = result["counterfactual_state"]
        changes = result.get("changed_features", [])
        changed_names = {c["feature"] for c in changes}

        if changes:
            for c in changes:
                f = c["feature"]
                orig_v = c["original_value"]
                cf_v   = c["counterfactual_value"]
                delta  = c["delta"]
                unit   = FEATURE_UNITS.get(f, "")
                sign   = "+" if delta > 0 else ""
                st.markdown(
                    f'<div class="sm-cf-block">'
                    f'<div class="sm-cf-feat">{FEATURE_LABELS.get(f, f)}</div>'
                    f'<div class="sm-cf-vals">'
                    f'<div><div class="sm-cf-orig">{orig_v:g}</div>'
                    f'<div style="font-family:var(--fm);font-size:11px;color:var(--gray);">ORIGINAL {unit}</div></div>'
                    f'<div class="sm-cf-arrow">\u2192</div>'
                    f'<div><div class="sm-cf-new">{cf_v:g}</div>'
                    f'<div style="font-family:var(--fm);font-size:11px;color:var(--gray);">SUGGESTED {unit}</div></div>'
                    f'</div>'
                    f'<div class="sm-cf-delta">{sign}{delta:g} {unit}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<p class="sm-body">No features were changed.</p>', unsafe_allow_html=True)

        with st.expander("FULL FEATURE COMPARISON"):
            trows = []
            for col in DETECTOR_FEATURES:
                orig_v = float(df_input[col].iloc[0])
                cf_v   = float(cf_state[col].iloc[0])
                if col in changed_names:
                    policy = "CHANGED"
                    style = "color:var(--blue);font-weight:700;"
                elif col in mutable_features:
                    policy = "MUTABLE / UNCHANGED"
                    style = ""
                else:
                    policy = "LOCKED / UNCHANGED"
                    style = "color:var(--gray);"
                trows.append(
                    f'<tr>'
                    f'<td style="font-family:var(--fm);font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:var(--gray);padding:.5rem 0;border-bottom:1px solid var(--divider);">{FEATURE_LABELS.get(col,col)}</td>'
                    f'<td style="font-family:var(--fm);font-size:13px;padding:.5rem .5rem;border-bottom:1px solid var(--divider);">{orig_v:g}</td>'
                    f'<td style="font-family:var(--fm);font-size:13px;padding:.5rem .5rem;border-bottom:1px solid var(--divider);">{cf_v:g}</td>'
                    f'<td style="font-family:var(--fm);font-size:11px;padding:.5rem 0;border-bottom:1px solid var(--divider);{style}">{policy}</td>'
                    f'</tr>'
                )
            st.markdown(
                '<table style="width:100%;border-collapse:collapse;">'
                '<tr style="border-bottom:1px solid var(--ink);">'
                '<th style="font-family:var(--fm);font-size:10px;text-transform:uppercase;letter-spacing:.07em;padding:.5rem 0;text-align:left;">FEATURE</th>'
                '<th style="font-family:var(--fm);font-size:10px;text-transform:uppercase;letter-spacing:.07em;padding:.5rem .5rem;text-align:left;">ORIGINAL</th>'
                '<th style="font-family:var(--fm);font-size:10px;text-transform:uppercase;letter-spacing:.07em;padding:.5rem .5rem;text-align:left;">SUGGESTED</th>'
                '<th style="font-family:var(--fm);font-size:10px;text-transform:uppercase;letter-spacing:.07em;padding:.5rem 0;text-align:left;">POLICY</th>'
                '</tr>' + "".join(trows) + '</table>',
                unsafe_allow_html=True,
            )

        # 3. VALIDATION
        section_heading("VALIDATION")
        feas_st  = result.get("feasibility_status", "unknown")
        plaus_st = result.get("plausibility_status", "unknown")
        feas_cls  = "sm-status-ok" if feas_st  == "feasible"   else "sm-status-fail"
        plaus_cls = "sm-status-ok" if plaus_st == "plausible" else "sm-status-fail"

        st.markdown(
            f'<div class="sm-status-row">'
            f'<div class="sm-status-cell"><div class="sm-label">VALIDITY</div><div class="sm-status-ok">VALID</div></div>'
            f'<div class="sm-status-cell"><div class="sm-label">FEASIBILITY</div><div class="{feas_cls}">{feas_st.upper()}</div></div>'
            f'<div class="sm-status-cell"><div class="sm-label">PLAUSIBILITY</div><div class="{plaus_cls}">{plaus_st.upper()}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if plaus_st == "implausible":
            st.markdown(
                '<div style="border-left:3px solid var(--red);padding:.75rem 1rem;margin:1rem 0;'
                'font-family:var(--fb);font-size:14px;color:var(--ink);">'
                '<strong>IMPLAUSIBLE</strong><br>'
                'The candidate is valid under the frozen detector and feasibility rules, '
                'but lies outside the configured normal-training distribution threshold. '
                'Treat this as a weak model-based suggestion requiring engineering review.'
                '</div>',
                unsafe_allow_html=True,
            )

        proximity = result.get("normalized_proximity", 0)
        gen_time  = result.get("generation_time_seconds", 0)
        plaus_dist = result.get("plausibility_distance", 0)
        plaus_thr  = result.get("plausibility_threshold", 0)
        st.markdown(
            f'<div class="sm-metrics-row" style="margin-top:1.5rem;">'
            f'<div class="sm-metric"><div class="sm-metric-label">SPARSITY</div><div class="sm-metric-value">{result.get("sparsity","—")}</div></div>'
            f'<div class="sm-metric"><div class="sm-metric-label">PROXIMITY</div><div class="sm-metric-value">{proximity:.4f}</div></div>'
            f'<div class="sm-metric"><div class="sm-metric-label">CANDIDATES</div><div class="sm-metric-value">{result.get("candidates_evaluated","—")}</div></div>'
            f'<div class="sm-metric"><div class="sm-metric-label">GENERATION</div><div class="sm-metric-value">{gen_time:.3f} s</div></div>'
            f'<div class="sm-metric"><div class="sm-metric-label">MAHL. DIST</div><div class="sm-metric-value">{f"{plaus_dist:.4f}" if plaus_dist else "—"}</div></div>'
            f'<div class="sm-metric"><div class="sm-metric-label">THRESHOLD</div><div class="sm-metric-value">{f"{plaus_thr:.4f}" if plaus_thr else "—"}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # 4. INTERPRETATION
        section_heading("INTERPRETATION")
        if changes:
            parts = []
            for c in changes:
                f = c["feature"]
                d = c["delta"]
                direction = "lower" if d < 0 else "higher"
                parts.append(f"a {direction} {FEATURE_LABELS.get(f, f).lower()}")
            interp = ", and ".join(parts)
            st.markdown(
                f'<div class="sm-interpretation">'
                f'A machine state with {interp} produced a nearby state that the frozen anomaly detector classified as normal.'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<p class="sm-body">No interpretable feature change was identified.</p>', unsafe_allow_html=True)

    else:
        # No counterfactual found
        reason = result.get("reason", "unknown").replace("_", " ").upper()
        n_cands = result.get("candidates_evaluated", "\u2014")
        gen_t   = result.get("generation_time_seconds", 0)
        st.markdown(
            f'<div style="margin:1.5rem 0;">'
            f'<div style="font-family:var(--fm);font-size:22px;font-weight:700;color:var(--ink);">NO VALID COUNTERFACTUAL FOUND</div>'
            f'<p class="sm-body" style="margin-top:.75rem;">No valid counterfactual was found within the configured search space and budget.</p>'
            f'<p class="sm-body" style="color:var(--gray);font-size:14px;">This does not imply that no real-world resolution exists.</p>'
            f'</div>'
            f'<div class="sm-metrics-row">'
            f'<div class="sm-metric"><div class="sm-metric-label">SEARCH STATUS</div><div class="sm-metric-value" style="font-size:14px;">{reason}</div></div>'
            f'<div class="sm-metric"><div class="sm-metric-label">CANDIDATES</div><div class="sm-metric-value">{n_cands}</div></div>'
            f'<div class="sm-metric"><div class="sm-metric-label">GENERATION TIME</div><div class="sm-metric-value">{gen_t:.3f} s</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # 5. LIMITATIONS
    section_heading("MODEL LIMITATIONS")
    st.markdown(
        '<div class="sm-notice">This counterfactual is:<ul>'
        '<li>Model-based, not causal.</li>'
        '<li>Not a certified safety recommendation.</li>'
        '<li>Not a guarantee of repair or failure prevention.</li>'
        '<li>Based on observed dataset bounds, which are not physical safety limits.</li>'
        '</ul>A valid, feasible counterfactual may still be implausible. '
        'The model is a frozen Isolation Forest baseline with limited recall.</div>',
        unsafe_allow_html=True,
    )

    # 6. DOWNLOADS
    section_heading("DOWNLOAD REPORT")
    r2 = dict(result)
    r2["original_state"] = result["original_state"].to_dict(orient="records")[0]
    if result.get("counterfactual_state") is not None and isinstance(result["counterfactual_state"], pd.DataFrame):
        r2["counterfactual_state"] = result["counterfactual_state"].to_dict(orient="records")[0]

    json_str = json.dumps(r2, indent=2, default=str)
    html_str = get_html_report(r2)

    dl1, dl2, dl3 = st.columns([3, 3, 4])
    with dl1:
        st.download_button("DOWNLOAD REPORT (HTML)", data=html_str,
            file_name="explainability_report.html", mime="text/html", use_container_width=True)
    with dl2:
        st.download_button("DOWNLOAD RESULT (JSON)", data=json_str,
            file_name="result.json", mime="application/json", use_container_width=True)

    st.markdown('<hr class="sm-rule">', unsafe_allow_html=True)
    na1, na2, _ = st.columns([2, 2, 4])
    with na1:
        if st.button("NEW ANALYSIS \u2192", key="btn_new", width="stretch"):
            navigate_to("single")
    with na2:
        if st.button("DASHBOARD", key="btn_to_dash", width="stretch"):
            navigate_to("dashboard")

# ═══════════════════════════════════════════════════════════════════
# ANALYTICS
# ═══════════════════════════════════════════════════════════════════
def view_analytics():
    render_header()
    st.markdown(
        '<h1 style="font-size:44px;line-height:1.02;letter-spacing:-.025em;margin-bottom:.25rem;">EVALUATION / 001</h1>'
        '<p class="sm-body" style="margin-bottom:2.5rem;">Pre-generated evaluation evidence from the offline report generator.</p>',
        unsafe_allow_html=True,
    )
    results_dir = PROJECT_ROOT / "results"
    req = {
        "detector_metrics": results_dir / "metrics" / "detector_metrics.json",
        "cf_metrics":       results_dir / "metrics" / "counterfactual_metrics.json",
        "confusion_matrix": results_dir / "plots"   / "detector_confusion_matrix.png",
        "cf_breakdown":     results_dir / "plots"   / "counterfactual_outcome_breakdown.png",
        "feat_freq":        results_dir / "plots"   / "changed_feature_frequency.png",
        "plaus_dist":       results_dir / "plots"   / "plausibility_distance_distribution.png",
        "gen_time":         results_dir / "plots"   / "generation_time_distribution.png",
    }
    missing = [k for k, p in req.items() if not p.exists()]
    if missing:
        section_heading("RESULTS UNAVAILABLE")
        st.markdown('<p class="sm-body">Evidence artifacts are not yet generated. Run the offline report generator:</p>', unsafe_allow_html=True)
        st.code("python tests/manual_generate_results.py")
        st.markdown(f'<p class="sm-body" style="font-size:13px;color:var(--gray);">Missing: {", ".join(missing)}</p>', unsafe_allow_html=True)
        return

    try:
        with open(req["detector_metrics"]) as f:
            det = json.load(f)
        with open(req["cf_metrics"]) as f:
            cf = json.load(f)
    except Exception as e:
        st.error(f"Error reading metrics: {e}")
        return

    section_heading("DETECTOR BASELINE")
    st.markdown(
        f'<div class="sm-metrics-row">'
        f'<div class="sm-metric"><div class="sm-metric-label">PRECISION</div><div class="sm-value">{det.get("precision",0)*100:.2f}%</div></div>'
        f'<div class="sm-metric"><div class="sm-metric-label">RECALL</div><div class="sm-value">{det.get("recall",0)*100:.2f}%</div></div>'
        f'<div class="sm-metric"><div class="sm-metric-label">F1</div><div class="sm-value">{det.get("f1_score",0)*100:.2f}%</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    section_heading("COUNTERFACTUAL EVALUATION")
    total_eval = cf.get("total_evaluated", cf.get("found", 0))
    found    = cf.get("found", 0)
    feasible = cf.get("feasible", 0)
    plausible = cf.get("plausible", 0)
    st.markdown(
        f'<div class="sm-metrics-row">'
        f'<div class="sm-metric"><div class="sm-metric-label">FOUND</div><div class="sm-value">{found} / {total_eval}</div></div>'
        f'<div class="sm-metric"><div class="sm-metric-label">FEASIBLE</div><div class="sm-value">{feasible} / {found}</div></div>'
        f'<div class="sm-metric"><div class="sm-metric-label">PLAUSIBLE</div><div class="sm-value">{plausible} / {found}</div></div>'
        f'</div>'
        f'<p class="sm-body" style="font-size:12px;color:var(--gray);margin-top:.5rem;">* Conditional on known machine failures detected as anomalous by the baseline model.</p>',
        unsafe_allow_html=True,
    )

    section_heading("EVIDENCE PLOTS")
    p1, p2 = st.columns(2, gap="large")
    with p1:
        st.image(str(req["confusion_matrix"]), caption="Detector Confusion Matrix",    use_container_width=True)
        st.image(str(req["feat_freq"]),        caption="Changed Feature Frequency",     use_container_width=True)
        st.image(str(req["gen_time"]),          caption="Generation Time Distribution", use_container_width=True)
    with p2:
        st.image(str(req["cf_breakdown"]),     caption="Counterfactual Outcome Breakdown",    use_container_width=True)
        st.image(str(req["plaus_dist"]),        caption="Plausibility Distance Distribution", use_container_width=True)

    section_heading("MODEL LIMITATIONS")
    st.markdown(
        '<div class="sm-notice"><ul>'
        '<li>Counterfactuals are model-based explanations, not causal root-cause conclusions.</li>'
        '<li>Suggested changes are not certified-safe operating instructions.</li>'
        '<li>Empirical feature bounds are not physical safety limits.</li>'
        '<li>A valid, feasible counterfactual may still be implausible.</li>'
        '<li>Model: frozen Isolation Forest baseline with limited recall.</li>'
        '</ul></div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="AnomalyCF",
        page_icon="\u2295",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    initialize_state()
    load_css()

    view = st.session_state.current_view
    if   view == "welcome":     view_welcome()
    elif view == "dashboard":   view_dashboard()
    elif view == "single":      view_single()
    elif view == "batch":       view_batch()
    elif view == "explanation": view_explanation()
    elif view == "analytics":   view_analytics()
    else:                       view_welcome()

if __name__ == "__main__":
    main()
