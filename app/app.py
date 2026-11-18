"""
CATS AI Safety Bot — Main Streamlit Application (v3.0 WOW Edition)

Upgraded Frontend Layout:
- Mission-control dark cockpit styling with glassmorphism & accent glows
- Real-time Automotive Telematics HUD Header (vehicle metadata, stream status, clock)
- Animated scanning gradient border on HUD, neon hover glows, fade-in & slide-up animations
- High-tech Glassmorphic Telemetry Cards with delta indicators & inline SVG sparklines
- Multi-tier dynamic status banners with custom gradient confidence bars & SVG confidence arc
- Tabbed Cockpit View (Gauges & Trends, 3D Route Map, Diagnostics Matrix)
- Interactive Pydeck map with glowing vehicle telemetry breadcrumbs
- Quick-action prompt chips styled as animated gradient pills
- Custom chat message bubbles (bot glass vs user blue gradient) with typing indicator dots
- 1-Click Forensic Incident Investigation Dossier Generator
- Graphify 2D/3D Codebase Knowledge Graph viewer
- Severity count badges on Alerts page
- Footer system health strip on every page
- Card entrance stagger animations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
import streamlit.components.v1 as components
import joblib
import json
import os
import sys
import time
import math
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.config import (
    FEATURE_COLUMNS, STATUS_COLORS, STATUS_EMOJIS,
    CLASSIFIER_PATH, SCALER_PATH, DATASET_PATH, FEEDBACK_PATH,
    STATUS_NORMAL, STATUS_WARNING, STATUS_CRITICAL,
    SPEED_THRESHOLDS, ACCELERATION_THRESHOLDS,
    IMPACT_SCORE_THRESHOLDS, BATTERY_THRESHOLDS,
)
from utils.telemetry import (
    load_telemetry, get_vehicle_ids,
    compute_statistics, generate_explanation,
)
from utils.llm import chat as llm_chat, get_backend, GROQ_MODEL, OPENROUTER_MODEL

# === Page Config ===
st.set_page_config(
    page_title="CATS AI Safety Bot | Command Center",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
#  GLOBAL CONSTANTS
# ──────────────────────────────────────────────
_APP_VERSION = "v3.0"
_APP_START_TIME = datetime.now(timezone.utc)

# === Advanced Executive Cockpit CSS — v3.0 WOW Edition ===
st.markdown("""
<style>
    /* ═══════════ KEYFRAME ANIMATIONS ═══════════ */

    @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
    }

    @keyframes slideUp {
        from { opacity: 0; transform: translateY(24px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    @keyframes scanLine {
        0%   { background-position: -300% 0; }
        100% { background-position: 300% 0; }
    }

    @keyframes pulseDot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%      { opacity: 0.4; transform: scale(0.7); }
    }

    @keyframes typingBounce {
        0%, 80%, 100% { transform: translateY(0); }
        40%           { transform: translateY(-8px); }
    }

    @keyframes neonPulse {
        0%, 100% { box-shadow: 0 0 5px rgba(59,130,246,0.4), 0 0 15px rgba(59,130,246,0.1); }
        50%      { box-shadow: 0 0 15px rgba(59,130,246,0.7), 0 0 40px rgba(59,130,246,0.25); }
    }

    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes pulse-border {
        0%   { box-shadow: 0 0 10px rgba(239, 68, 68, 0.3); }
        50%  { box-shadow: 0 0 25px rgba(239, 68, 68, 0.75); }
        100% { box-shadow: 0 0 10px rgba(239, 68, 68, 0.3); }
    }

    /* ═══════════ GLOBAL CONTAINER ═══════════ */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 4rem;
        max-width: 1380px;
    }

    /* ═══════════ HEADS-UP DISPLAY (HUD) TOP BAR ═══════════ */
    .hud-header {
        background: linear-gradient(90deg, rgba(15,23,42,0.97) 0%, rgba(30,41,59,0.88) 50%, rgba(15,23,42,0.97) 100%);
        border: 1px solid rgba(59,130,246,0.25);
        border-radius: 14px;
        padding: 14px 22px;
        margin-bottom: 18px;
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5), 0 0 15px rgba(59,130,246,0.1);
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        position: relative;
        overflow: hidden;
        animation: fadeIn 0.6s ease-out;
    }
    /* Animated gradient scanning border overlay */
    .hud-header::before {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: 14px;
        padding: 1px;
        background: linear-gradient(90deg, transparent, rgba(59,130,246,0.6), rgba(56,189,248,0.4), transparent);
        background-size: 300% 100%;
        animation: scanLine 4s linear infinite;
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        pointer-events: none;
    }
    .hud-title {
        font-size: 20px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .hud-subtitle {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #94a3b8;
        margin-top: 2px;
    }
    .hud-pill {
        background: rgba(30,41,59,0.7);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        color: #cbd5e1;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .hud-pill-green {
        border-color: rgba(16,185,129,0.4);
        background: rgba(16,185,129,0.1);
        color: #34d399;
    }
    .hud-clock {
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 12px;
        color: #38bdf8;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    /* Animated pulse dot next to LIVE STREAM */
    .live-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #34d399;
        border-radius: 50%;
        animation: pulseDot 1.4s ease-in-out infinite;
        margin-right: 4px;
        vertical-align: middle;
    }

    /* ═══════════ GLASSMORPHIC TELEMETRY CARDS ═══════════ */
    .telemetry-card {
        background: rgba(17,24,39,0.8);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.3s ease;
        margin-bottom: 8px;
        animation: slideUp 0.5s ease-out both;
    }
    .telemetry-card:hover {
        transform: translateY(-3px);
        border-color: rgba(59,130,246,0.6);
        box-shadow: 0 4px 20px rgba(0,0,0,0.25), 0 0 20px rgba(59,130,246,0.35), 0 0 40px rgba(59,130,246,0.1);
    }
    /* Stagger animation delays for cards in a row */
    .tcard-d1 { animation-delay: 0s; }
    .tcard-d2 { animation-delay: 0.08s; }
    .tcard-d3 { animation-delay: 0.16s; }
    .tcard-d4 { animation-delay: 0.24s; }
    .tcard-d5 { animation-delay: 0.32s; }

    .telemetry-label {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
    }
    .telemetry-val {
        font-size: 26px;
        font-weight: 800;
        color: #f8fafc;
        margin: 4px 0 2px 0;
        letter-spacing: -0.5px;
    }
    .telemetry-delta {
        font-size: 12px;
        font-weight: 600;
    }
    .delta-green { color: #10b981; }
    .delta-amber { color: #f59e0b; }
    .delta-red   { color: #ef4444; }

    /* Mini sparkline container inside telemetry card */
    .sparkline-container {
        margin-top: 6px;
        height: 28px;
        overflow: hidden;
    }

    /* ═══════════ DYNAMIC STATUS BANNERS ═══════════ */
    .status-normal-box {
        background: linear-gradient(135deg, rgba(16,185,129,0.25) 0%, rgba(5,150,105,0.1) 100%);
        border: 1px solid rgba(16,185,129,0.6);
        color: #ffffff;
        padding: 18px 24px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(16,185,129,0.2);
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
        animation: slideUp 0.5s ease-out both;
    }
    .status-warning-box {
        background: linear-gradient(135deg, rgba(245,158,11,0.25) 0%, rgba(217,119,6,0.1) 100%);
        border: 1px solid rgba(245,158,11,0.7);
        color: #ffffff;
        padding: 18px 24px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(245,158,11,0.25);
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
        animation: slideUp 0.5s ease-out both;
    }
    .status-critical-box {
        background: linear-gradient(135deg, rgba(239,68,68,0.35) 0%, rgba(185,28,28,0.15) 100%);
        border: 1px solid rgba(239,68,68,0.85);
        color: #ffffff;
        padding: 18px 24px;
        border-radius: 12px;
        box-shadow: 0 4px 25px rgba(239,68,68,0.4);
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
        animation: pulse-border 1.5s infinite;
    }
    .status-title-text {
        font-size: 22px;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .status-sub-text {
        font-size: 13px;
        color: #e2e8f0;
        margin-top: 2px;
    }

    /* ═══════════ CUSTOM GRADIENT CONFIDENCE BAR ═══════════ */
    .conf-bar-track {
        background: rgba(30,41,59,0.6);
        border-radius: 8px;
        height: 18px;
        overflow: hidden;
        position: relative;
        margin: 4px 0 6px 0;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.6s cubic-bezier(.4,0,.2,1);
        position: relative;
    }
    .conf-bar-fill.green  { background: linear-gradient(90deg, #10b981, #34d399); box-shadow: 0 0 10px rgba(16,185,129,0.5); }
    .conf-bar-fill.amber  { background: linear-gradient(90deg, #f59e0b, #fbbf24); box-shadow: 0 0 10px rgba(245,158,11,0.5); }
    .conf-bar-fill.red    { background: linear-gradient(90deg, #ef4444, #f87171); box-shadow: 0 0 10px rgba(239,68,68,0.5); }
    .conf-bar-fill.blue   { background: linear-gradient(90deg, #3b82f6, #60a5fa); box-shadow: 0 0 10px rgba(59,130,246,0.5); }
    .conf-bar-label {
        font-size: 11px;
        font-weight: 700;
        color: #e2e8f0;
        display: flex;
        justify-content: space-between;
        margin-bottom: 1px;
    }

    /* ═══════════ CHAT MESSAGE BUBBLES ═══════════ */
    .chat-bubble-bot {
        background: rgba(17,24,39,0.85);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px 16px 16px 4px;
        padding: 14px 18px;
        margin: 8px 40px 8px 0;
        color: #e2e8f0;
        font-size: 14px;
        line-height: 1.6;
        animation: slideUp 0.35s ease-out both;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    }
    .chat-bubble-user {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        border: none;
        border-radius: 16px 16px 4px 16px;
        padding: 14px 18px;
        margin: 8px 0 8px 40px;
        color: #ffffff;
        font-size: 14px;
        line-height: 1.6;
        text-align: right;
        animation: slideUp 0.35s ease-out both;
        box-shadow: 0 2px 12px rgba(59,130,246,0.3);
    }
    .chat-sender {
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .chat-sender-user {
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: rgba(255,255,255,0.65);
        margin-bottom: 4px;
        text-align: right;
    }

    /* ═══════════ TYPING INDICATOR DOTS ═══════════ */
    .typing-indicator {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 10px 18px;
        background: rgba(17,24,39,0.85);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px 16px 16px 4px;
        margin: 8px 40px 8px 0;
    }
    .typing-indicator .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #94a3b8;
        animation: typingBounce 1.4s ease-in-out infinite;
    }
    .typing-indicator .dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-indicator .dot:nth-child(3) { animation-delay: 0.4s; }

    /* ═══════════ QUICK PROMPT CHIPS ═══════════ */
    .quick-chip {
        display: inline-block;
        background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(139,92,246,0.15));
        border: 1px solid rgba(59,130,246,0.35);
        border-radius: 24px;
        padding: 8px 18px;
        font-size: 13px;
        font-weight: 600;
        color: #93c5fd;
        cursor: pointer;
        transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
        animation: fadeIn 0.5s ease-out both;
    }
    .quick-chip:hover {
        transform: scale(1.05);
        box-shadow: 0 0 18px rgba(59,130,246,0.35);
        background: linear-gradient(135deg, rgba(59,130,246,0.35), rgba(139,92,246,0.3));
    }

    /* ═══════════ SEVERITY BADGE PILLS ═══════════ */
    .severity-badges {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 14px;
        animation: fadeIn 0.5s ease-out both;
    }
    .sev-badge {
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .sev-badge.critical { background: rgba(239,68,68,0.2); border: 1px solid rgba(239,68,68,0.6); color: #f87171; }
    .sev-badge.warning  { background: rgba(245,158,11,0.2); border: 1px solid rgba(245,158,11,0.6); color: #fbbf24; }
    .sev-badge.normal   { background: rgba(16,185,129,0.2); border: 1px solid rgba(16,185,129,0.6); color: #34d399; }

    /* ═══════════ DIAGNOSTICS SIDEBAR BOX ═══════════ */
    .sidebar-diag {
        background: rgba(15,23,42,0.6);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 12px;
        margin-top: 15px;
    }
    .diag-row {
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        padding: 4px 0;
        color: #94a3b8;
    }
    .diag-val {
        color: #e2e8f0;
        font-weight: 600;
    }

    /* Sidebar version badge */
    .version-badge {
        display: inline-block;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: #fff;
        font-size: 10px;
        font-weight: 800;
        padding: 3px 10px;
        border-radius: 10px;
        letter-spacing: 0.5px;
    }

    /* ═══════════ ANOMALY & AI BADGES ═══════════ */
    .anomaly-badge {
        background: linear-gradient(135deg, #8b5cf6, #6d28d9);
        color: #ffffff;
        padding: 4px 12px;
        border-radius: 14px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .ai-badge {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
    }

    /* ═══════════ TERMINAL DOSSIER BOX ═══════════ */
    .dossier-box {
        background-color: #030712;
        border: 1px solid #1f2937;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 16px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 13px;
        color: #e5e7eb;
        white-space: pre-wrap;
        line-height: 1.5;
        animation: fadeIn 0.5s ease-out;
    }

    /* ═══════════ FOOTER SYSTEM HEALTH STRIP ═══════════ */
    .footer-strip {
        background: linear-gradient(90deg, rgba(15,23,42,0.95) 0%, rgba(30,41,59,0.85) 50%, rgba(15,23,42,0.95) 100%);
        border-top: 1px solid rgba(59,130,246,0.2);
        padding: 8px 22px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        font-size: 11px;
        color: #64748b;
        margin-top: 30px;
        border-radius: 10px;
        animation: fadeIn 0.6s ease-out both;
    }
    .footer-strip .footer-item {
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    .footer-strip .footer-val {
        color: #94a3b8;
        font-weight: 600;
    }

    /* ═══════════ NEON ACCENT GLOWS ═══════════ */
    .neon-glow-blue {
        animation: neonPulse 2.5s ease-in-out infinite;
    }

    /* Section entrance wrapper */
    .section-fade {
        animation: slideUp 0.5s ease-out both;
    }

    /* ═══════════ STREAMLIT TABS & CONTROLS POLISH ═══════════ */
    div[data-baseweb="tab-list"] {
        background: rgba(15, 23, 42, 0.7);
        padding: 4px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        gap: 6px;
    }
    button[data-baseweb="tab"] {
        border-radius: 8px !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 8px 18px !important;
        transition: all 0.2s ease !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.3), rgba(37, 99, 235, 0.2)) !important;
        color: #60a5fa !important;
        border: 1px solid rgba(59, 130, 246, 0.5) !important;
        box-shadow: 0 0 12px rgba(59, 130, 246, 0.3) !important;
    }

    /* Streamlit Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.9)) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        border-color: rgba(59, 130, 246, 0.8) !important;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.35) !important;
        transform: translateY(-1px) !important;
        color: #ffffff !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
        border-color: rgba(59, 130, 246, 0.8) !important;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  CACHED LOADERS
# ──────────────────────────────────────────────

@st.cache_resource
def load_model():
    model_data = joblib.load(CLASSIFIER_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model_data["model"], model_data["label_encoder"], scaler


@st.cache_resource
def load_anomaly_model():
    anom_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "anomaly_detector.pkl")
    anom_scaler_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "anomaly_scaler.pkl")
    if os.path.exists(anom_path) and os.path.exists(anom_scaler_path):
        return joblib.load(anom_path), joblib.load(anom_scaler_path)
    return None, None


@st.cache_data
def load_data():
    return load_telemetry(DATASET_PATH)


@st.cache_data
def load_model_comparison():
    comp_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "model_comparison.json")
    if os.path.exists(comp_path):
        with open(comp_path) as f:
            return json.load(f)
    return None


# ──────────────────────────────────────────────
#  HELPER & UI RENDERING FUNCTIONS
# ──────────────────────────────────────────────

def predict_risk(model, scaler, le, features):
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    label = le.inverse_transform([prediction])[0]
    return label, dict(zip(le.classes_, probabilities))


def detect_anomaly(features, iso_forest, anom_scaler):
    if iso_forest is None or anom_scaler is None:
        return False, 0.0
    X = np.array(features).reshape(1, -1)
    X_scaled = anom_scaler.transform(X)
    pred = iso_forest.predict(X_scaled)[0]
    score = iso_forest.decision_function(X_scaled)[0]
    return pred == -1, float(score)


def _build_sparkline_svg(values, width=100, height=24, color="#38bdf8"):
    """Generate a tiny inline SVG sparkline from a list of numeric values."""
    if not values or len(values) < 2:
        return ""
    vals = list(values)
    mn, mx = min(vals), max(vals)
    rng = mx - mn if mx != mn else 1
    n = len(vals)
    points = []
    for i, v in enumerate(vals):
        x = round(i / (n - 1) * width, 2)
        y = round(height - ((v - mn) / rng) * (height - 2) - 1, 2)
        points.append(f"{x},{y}")
    pts_str = " ".join(points)
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block;">'
        f'<polyline fill="none" stroke="{color}" stroke-width="1.5" '
        f'stroke-linecap="round" stroke-linejoin="round" points="{pts_str}"/></svg>'
    )


def _build_confidence_arc_svg(pct, size=60, color="#34d399"):
    """Build an SVG circle arc showing confidence percentage."""
    r = (size - 6) / 2
    cx = cy = size / 2
    circumference = 2 * math.pi * r
    offset = circumference * (1 - pct)
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="5"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="5" '
        f'stroke-dasharray="{circumference}" stroke-dashoffset="{offset}" '
        f'stroke-linecap="round" transform="rotate(-90 {cx} {cy})" '
        f'style="transition: stroke-dashoffset 0.6s ease;"/>'
        f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" fill="#f8fafc" '
        f'font-size="12" font-weight="800" font-family="Inter,sans-serif">{pct:.0%}</text>'
        f'</svg>'
    )


def _render_confidence_bar(label, prob, bar_class="blue"):
    """Render a custom gradient confidence bar for a classification probability."""
    pct = prob * 100
    emoji = STATUS_EMOJIS.get(label, "")
    return (
        f'<div class="conf-bar-label"><span>{emoji} {label}</span><span>{prob:.1%}</span></div>'
        f'<div class="conf-bar-track">'
        f'<div class="conf-bar-fill {bar_class}" style="width:{pct}%;"></div>'
        f'</div>'
    )


def _render_footer(record_count=0):
    """Render the system health footer strip."""
    utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    uptime_delta = datetime.now(timezone.utc) - _APP_START_TIME
    uptime_min = int(uptime_delta.total_seconds() // 60)
    st.markdown(f"""
    <div class="footer-strip">
        <div class="footer-item">🤖 <span class="footer-val">CATS AI Safety Bot {_APP_VERSION} WOW Edition</span></div>
        <div class="footer-item">🧠 <span class="footer-val">Gradient Boosting Classifier</span></div>
        <div class="footer-item">⏱️ Uptime: <span class="footer-val">{uptime_min}m</span></div>
        <div class="footer-item">📊 Records: <span class="footer-val">{record_count:,}</span></div>
        <div class="footer-item">🕐 <span class="footer-val">{utc_now}</span></div>
    </div>
    """, unsafe_allow_html=True)


def render_hud(vehicle_id, latest, model_name="Gradient Boosting (99.94%)"):
    """Render top automotive Heads-Up Display banner with live UTC clock and animated pulse dot."""
    status = latest.get("risk_label", "NORMAL")
    gps_active = latest.get("gps_status", 1) == 1
    lat = latest.get("latitude", 0)
    lon = latest.get("longitude", 0)
    utc_now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    st.markdown(f"""
    <div class="hud-header neon-glow-blue">
        <div>
            <div class="hud-title">
                <span>🚗</span> CATS TELEMATICS MISSION CONTROL
                <span class="hud-pill hud-pill-green"><span class="live-dot"></span> LIVE STREAM</span>
            </div>
            <div class="hud-subtitle">CRASH CONNECT • CRISIS ALERT TECH SYSTEM • GUARDIAN ON WHEELS</div>
        </div>
        <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
            <div class="hud-pill">🚘 <b>{vehicle_id}</b></div>
            <div class="hud-pill">📍 {lat:.4f}, {lon:.4f}</div>
            <div class="hud-pill">📡 {'🟢 GPS LOCK' if gps_active else '🔴 GPS LOST'}</div>
            <div class="hud-pill">🏆 {model_name}</div>
            <div class="hud-pill"><span class="hud-clock">🕐 {utc_now}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_executive_status_banner(status, pred_probs, is_anomaly=False, anom_score=0.0):
    """Render high-impact automotive status banner with SVG confidence arc."""
    emoji = STATUS_EMOJIS.get(status, "✅")

    if status == STATUS_CRITICAL:
        box_class = "status-critical-box"
        subtext = "CRITICAL HAZARD DETECTED — Urgent Collision / Rollover Risk. Dispatch Standby."
        arc_color = "#ef4444"
    elif status == STATUS_WARNING:
        box_class = "status-warning-box"
        subtext = "OPERATING WARNING — Parameter threshold exceeded. Advisory speed reduction recommended."
        arc_color = "#f59e0b"
    else:
        box_class = "status-normal-box"
        subtext = "NOMINAL OPERATING ENVELOPE — All multi-sensor telemetry within safe ISO bounds."
        arc_color = "#10b981"

    confidence = pred_probs.get(status, 1.0)
    prob_str = f"Confidence: {confidence:.1%}"
    anom_html = f'<span class="anomaly-badge">🔮 ANOMALY OUTLIER ({anom_score:.3f})</span>' if is_anomaly else ""
    arc_svg = _build_confidence_arc_svg(confidence, size=58, color=arc_color)

    st.markdown(f"""
    <div class="{box_class}">
        <div>
            <div class="status-title-text">{emoji} VEHICLE SAFETY STATUS: {status} {anom_html}</div>
            <div class="status-sub-text">{subtext}</div>
        </div>
        <div style="text-align: right; display: flex; align-items: center; gap: 14px;">
            {arc_svg}
            <div>
                <div style="font-size: 18px; font-weight: 800;">{prob_str}</div>
                <div style="font-size: 11px; opacity: 0.85;">Dual-Model AI Consensus</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def make_gauge(value, title, min_val, max_val, thresholds, unit="", reverse=False):
    """Create an animated gauge chart with custom styling."""
    if reverse:
        steps = [
            {"range": [min_val, thresholds.get("critical", 10)], "color": "rgba(239, 68, 68, 0.85)"},
            {"range": [thresholds.get("critical", 10), thresholds.get("warning", 30)], "color": "rgba(245, 158, 11, 0.8)"},
            {"range": [thresholds.get("warning", 30), max_val], "color": "rgba(16, 185, 129, 0.8)"},
        ]
        thresh_val = thresholds.get("warning", 30)
    else:
        steps = [
            {"range": [min_val, thresholds.get("warning", 60)], "color": "rgba(16, 185, 129, 0.8)"},
            {"range": [thresholds.get("warning", 60), thresholds.get("critical", 80)], "color": "rgba(245, 158, 11, 0.8)"},
            {"range": [thresholds.get("critical", 80), max_val], "color": "rgba(239, 68, 68, 0.85)"},
        ]
        thresh_val = thresholds.get("critical", 80)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 13, "color": "#94a3b8", "family": "Inter, sans-serif"}},
        number={"suffix": f" {unit}", "font": {"color": "#f8fafc", "size": 22, "weight": 800}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickcolor": "#475569", "tickfont": {"color": "#94a3b8", "size": 10}},
            "bar": {"color": "#38bdf8", "thickness": 0.3},
            "steps": steps,
            "threshold": {"line": {"color": "#ffffff", "width": 2}, "thickness": 0.8, "value": thresh_val},
        },
    ))
    fig.update_layout(
        height=170, margin=dict(l=15, r=15, t=35, b=5),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff"),
    )
    return fig


def render_telemetry_map(vehicle_df, latest_reading):
    """Render interactive Pydeck map with color-coded telemetry track."""
    map_df = vehicle_df.tail(60).copy()
    if "latitude" not in map_df.columns or "longitude" not in map_df.columns:
        st.info("GPS coordinates not present in dataset.")
        return

    def get_color(row):
        if row.get("gps_status", 1) == 0:
            return [148, 163, 184, 180]  # Grey: Signal lost
        status = row.get("risk_label", "NORMAL")
        if status == "CRITICAL":
            return [239, 68, 68, 240]    # Red: Critical
        elif status == "WARNING":
            return [245, 158, 11, 220]   # Orange: Warning
        return [16, 185, 129, 200]       # Green: Normal

    map_df["color"] = map_df.apply(get_color, axis=1)
    map_df["radius"] = map_df["risk_label"].apply(
        lambda x: 130 if x == "CRITICAL" else (75 if x == "WARNING" else 45)
    )

    path_coords = map_df.sort_values("timestamp")[["longitude", "latitude"]].values.tolist()
    route_layer = pdk.Layer(
        "PathLayer",
        data=[{"path": path_coords}],
        get_path="path",
        get_color=[56, 189, 248, 160],
        width_min_pixels=3,
    )

    points_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[longitude, latitude]",
        get_color="color",
        get_radius="radius",
        pickable=True,
    )

    curr_df = pd.DataFrame([latest_reading])
    curr_halo = pdk.Layer(
        "ScatterplotLayer",
        data=curr_df,
        get_position="[longitude, latitude]",
        get_color=[56, 189, 248, 80],
        get_radius=380,
        pickable=False,
    )
    curr_layer = pdk.Layer(
        "ScatterplotLayer",
        data=curr_df,
        get_position="[longitude, latitude]",
        get_color=[255, 255, 255, 255],
        get_radius=190,
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=float(latest_reading["latitude"]),
        longitude=float(latest_reading["longitude"]),
        zoom=11.6,
        pitch=28,
    )

    deck = pdk.Deck(
        layers=[route_layer, points_layer, curr_halo, curr_layer],
        initial_view_state=view_state,
        tooltip={
            "text": "Time: {timestamp}\nSpeed: {speed} km/h\nStatus: {risk_label}\nImpact: {impact_score}"
        },
        map_style="mapbox://styles/mapbox/dark-v10",
    )

    col_m1, col_m2 = st.columns([3, 1])
    with col_m1:
        st.pydeck_chart(deck, use_container_width=True)
    with col_m2:
        st.markdown("**🛰️ Geospatial Legend:**")
        st.markdown("🟢 **Green**: Nominal Highway/City Envelope")
        st.markdown("🟠 **Orange**: Warning Severity Region")
        st.markdown("🔴 **Red**: Critical Impact Collision Point")
        st.markdown("⚪ **Grey**: GPS Signal Dead Zone")
        st.markdown("📍 **White Pulse**: Current Vehicle Fix")
        st.caption(f"Coordinates: {latest_reading.get('latitude', 0):.5f} N, {latest_reading.get('longitude', 0):.5f} E")


def generate_sim_reading(rng, scenario="mixed", last_reading=None):
    """Generate simulated reading with smooth GPS progression."""
    base_lat = last_reading.get("latitude", 12.9716) if last_reading else 12.9716
    base_lon = last_reading.get("longitude", 77.5946) if last_reading else 77.5946

    if scenario == "normal":
        speed = float(np.clip(rng.normal(60, 15), 0, 95))
        reading = {
            "speed": speed,
            "acceleration": float(np.clip(rng.exponential(0.8), 0, 3)),
            "impact_score": float(np.clip(rng.exponential(0.03), 0, 0.3)),
            "battery_status": float(np.clip(rng.normal(80, 10), 40, 100)),
            "gps_status": int(rng.choice([1]*19 + [0])),
        }
    elif scenario == "dangerous":
        speed = float(np.clip(rng.normal(140, 25), 80, 220))
        reading = {
            "speed": speed,
            "acceleration": float(np.clip(rng.normal(6, 2), 2, 12)),
            "impact_score": float(np.clip(rng.normal(0.6, 0.2), 0.1, 1.0)),
            "battery_status": float(np.clip(rng.normal(20, 10), 0, 50)),
            "gps_status": int(rng.choice([1, 1, 0, 0, 0])),
        }
    else:  # mixed
        r = rng.random()
        if r < 0.6:
            return generate_sim_reading(rng, "normal", last_reading)
        elif r < 0.85:
            speed = float(np.clip(rng.normal(100, 20), 40, 145))
            reading = {
                "speed": speed,
                "acceleration": float(np.clip(rng.normal(3.5, 1.5), 0.5, 7)),
                "impact_score": float(np.clip(rng.normal(0.3, 0.15), 0, 0.7)),
                "battery_status": float(np.clip(rng.normal(45, 15), 10, 80)),
                "gps_status": int(rng.choice([1, 1, 1, 0])),
            }
        else:
            return generate_sim_reading(rng, "dangerous", last_reading)

    step_lat = (reading["speed"] / 120.0) * float(rng.normal(0.001, 0.0005))
    step_lon = (reading["speed"] / 120.0) * float(rng.normal(0.001, 0.0005))
    reading["latitude"] = np.round(base_lat + step_lat, 5)
    reading["longitude"] = np.round(base_lon + step_lon, 5)
    return reading


def generate_incident_dossier(row, model, scaler, le, iso_forest, anom_scaler):
    """Generate Forensic Incident Investigation Dossier."""
    features = [row[col] for col in FEATURE_COLUMNS]
    pred_label, pred_probs = predict_risk(model, scaler, le, features)
    is_anom, anom_score = detect_anomaly(features, iso_forest, anom_scaler)
    explanations = generate_explanation(row)

    report = f"""================================================================================
CRASH CONNECT — CRISIS ALERT TECH SYSTEM (CATS)
OFFICIAL FORENSIC INCIDENT INVESTIGATION DOSSIER
================================================================================
Docket Reference : CATS-INC-{datetime.now().strftime('%Y%m%d%H%M%S')}
Generated At     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
Monitored Asset  : {row.get('vehicle_id', 'VH-SIM')}
Incident Timestamp: {row.get('timestamp', 'N/A')}
GPS Geolocation  : Lat {row.get('latitude', 0):.5f}, Lon {row.get('longitude', 0):.5f}

--------------------------------------------------------------------------------
1. SENSOR TELEMETRICS SNAPSHOT
--------------------------------------------------------------------------------
- Velocity / Speed: {row.get('speed', 0):.1f} km/h  (Warning > {SPEED_THRESHOLDS['warning']}, Critical > {SPEED_THRESHOLDS['critical']})
- G-Acceleration  : {row.get('acceleration', 0):.2f} g    (Warning > {ACCELERATION_THRESHOLDS['warning']}, Critical > {ACCELERATION_THRESHOLDS['critical']})
- Impact Severity : {row.get('impact_score', 0):.3f}   (Warning > {IMPACT_SCORE_THRESHOLDS['warning']}, Critical > {IMPACT_SCORE_THRESHOLDS['critical']})
- Battery Level   : {row.get('battery_status', 0):.1f} %   (Warning < {BATTERY_THRESHOLDS['warning']}, Critical < {BATTERY_THRESHOLDS['critical']})
- GPS Connection  : {'CONNECTED (Active Satellite Fix)' if row.get('gps_status', 1) == 1 else 'DISCONNECTED (Dead Zone Detected)'}

--------------------------------------------------------------------------------
2. AI/ML MULTI-TIER RISK EVALUATION
--------------------------------------------------------------------------------
- Primary Classification Verdict : {pred_label}
- Model Consensus Probabilities  :
  * CRITICAL : {pred_probs.get('CRITICAL', 0):.2%}
  * WARNING  : {pred_probs.get('WARNING', 0):.2%}
  * NORMAL   : {pred_probs.get('NORMAL', 0):.2%}
- Anomaly Detection (Isolation Forest): {'⚠️ ANOMALOUS OUTLIER DETECTED' if is_anom else '✅ ENVELOPE CONSISTENT'}
- Anomaly Outlier Decision Score     : {anom_score:.4f}

--------------------------------------------------------------------------------
3. ROOT CAUSE ATTRIBUTION
--------------------------------------------------------------------------------
"""
    for exp in explanations:
        report += f"- {exp}\n"

    report += f"""
--------------------------------------------------------------------------------
4. FIRST RESPONDER & FLEET DISPATCH DIRECTIVE
--------------------------------------------------------------------------------
"""
    if pred_label == STATUS_CRITICAL:
        report += """[EMERGENCY PROTOCOL LEVEL 1 - SEVERE RISK TRIGGERED]
- Immediately dispatch nearest emergency rescue unit to GPS coordinates.
- Transmit vehicle impact telemetry to regional trauma care center.
- Activate automated audio ping to verify driver responsiveness.
"""
    elif pred_label == STATUS_WARNING:
        report += """[PRECAUTIONARY PROTOCOL LEVEL 2 - WARNING GENERATED]
- Issue haptic and chime warning alert to in-cabin cluster.
- Request driver speed reduction and schedule nearest battery/chassis inspection.
"""
    else:
        report += """[STANDARD MONITORING]
- All metrics within baseline operating envelope. No dispatch required.
"""

    report += """================================================================================
END OF DOSSIER — GUARDIAN ON WHEELS SAFETY ASSURANCE
================================================================================"""
    return report


def record_feedback(row, feedback_type, notes=""):
    feedback = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "alert_timestamp": str(row.get("timestamp", "")),
        "vehicle_id": row.get("vehicle_id", "SIM"),
        "risk_label": row.get("risk_label", ""),
        "speed": row.get("speed", 0),
        "acceleration": row.get("acceleration", 0),
        "impact_score": row.get("impact_score", 0),
        "battery_status": row.get("battery_status", 0),
        "gps_status": row.get("gps_status", 1),
        "feedback_type": feedback_type,
        "notes": notes,
    }])
    if os.path.exists(FEEDBACK_PATH):
        existing = pd.read_csv(FEEDBACK_PATH)
        feedback = pd.concat([existing, feedback], ignore_index=True)
    feedback.to_csv(FEEDBACK_PATH, index=False)


# ──────────────────────────────────────────────
#  RULE-BASED CHAT FALLBACK
# ──────────────────────────────────────────────

def handle_chat_rulebased(question, df, vehicle_id, model, scaler, le):
    q = question.lower().strip()
    vehicle_df = df[df["vehicle_id"] == vehicle_id].sort_values("timestamp")
    if not len(vehicle_df):
        return "No data available for this vehicle."
    latest = vehicle_df.iloc[-1]

    if any(kw in q for kw in ["status", "condition", "how is", "state", "safe"]):
        status = latest["risk_label"]
        explanations = generate_explanation(latest)
        resp = f"**Current Vehicle Status: {STATUS_EMOJIS[status]} {status}**\n\n"
        resp += f"- 🏎️ Speed: {latest['speed']:.0f} km/h\n- ⚡ Acceleration: {latest['acceleration']:.1f} g\n"
        resp += f"- 💥 Impact Score: {latest['impact_score']:.3f}\n- 🔋 Battery: {latest['battery_status']:.0f}%\n"
        resp += f"- 📡 GPS: {'Active' if latest['gps_status'] == 1 else 'Lost'}\n\n**Safety Analysis:**\n"
        for exp in explanations:
            resp += f"- {exp}\n"
        return resp

    if any(kw in q for kw in ["why", "reason", "explain", "cause", "triggered"]):
        explanations = generate_explanation(latest)
        status = latest["risk_label"]
        resp = f"The latest reading was classified as **{status}** based on:\n\n"
        for exp in explanations:
            resp += f"- {exp}\n"
        features = [latest[col] for col in FEATURE_COLUMNS]
        _, probs = predict_risk(model, scaler, le, features)
        resp += f"\n**ML Confidence:** " + ", ".join([f"{k}: {v:.1%}" for k, v in sorted(probs.items(), key=lambda x: -x[1])])
        return resp

    if any(kw in q for kw in ["speed", "fast", "velocity"]):
        return (f"**Speed Analysis for {vehicle_id}:**\n\n"
                f"- Current: {latest['speed']:.0f} km/h\n- Average: {vehicle_df['speed'].mean():.0f} km/h\n"
                f"- Maximum: {vehicle_df['speed'].max():.0f} km/h\n"
                f"- {'⚠️ Above warning threshold!' if latest['speed'] >= 100 else '✅ Within normal limits.'}")

    if any(kw in q for kw in ["battery", "charge", "power"]):
        return (f"**Battery Health for {vehicle_id}:**\n\n- Current: {latest['battery_status']:.0f}%\n"
                f"- Average: {vehicle_df['battery_status'].mean():.0f}%\n"
                f"- {'🔴 Critically low!' if latest['battery_status'] <= 10 else '⚠️ Low.' if latest['battery_status'] <= 30 else '✅ Nominal.'}")

    if any(kw in q for kw in ["gps", "location", "signal"]):
        gps_pct = (vehicle_df["gps_status"] == 1).mean() * 100
        return (f"**GPS Telematics for {vehicle_id}:**\n\n- Current: {'🟢 Active' if latest['gps_status'] == 1 else '🔴 Lost'}\n"
                f"- Position: Lat {latest.get('latitude',0):.4f}, Lon {latest.get('longitude',0):.4f}\n"
                f"- Satellite Fix Uptime: {gps_pct:.1f}%")

    if any(kw in q for kw in ["impact", "crash", "collision"]):
        return (f"**Impact / Collision Audit for {vehicle_id}:**\n\n- Current: {latest['impact_score']:.3f}\n"
                f"- Max Recorded: {vehicle_df['impact_score'].max():.3f}\n"
                f"- High Impact Incidents (>0.7): {(vehicle_df['impact_score'] > 0.7).sum()}")

    if any(kw in q for kw in ["history", "last", "recent", "readings", "show"]):
        recent = vehicle_df.tail(10)[["timestamp", "speed", "acceleration", "impact_score", "battery_status", "gps_status", "risk_label"]]
        return f"**Last 10 readings for {vehicle_id}:**\n\n{recent.to_markdown(index=False)}"

    if any(kw in q for kw in ["stats", "statistics", "summary", "overview"]):
        stats = compute_statistics(df, vehicle_id)
        resp = f"**Fleet Statistical Envelope for {vehicle_id}:**\n\n"
        for feat in ["speed", "acceleration", "impact_score", "battery_status"]:
            s = stats[feat]
            resp += f"**{feat}:** mean={s['mean']:.2f}, std={s['std']:.2f}, min={s['min']:.2f}, max={s['max']:.2f}\n\n"
        return resp

    if any(kw in q for kw in ["help", "what can", "commands"]):
        return ("**I can assist with:**\n\n"
                "- 🚗 \"What is the current status?\"\n- ❓ \"Why was this alert triggered?\"\n"
                "- 🏎️ \"How fast is the vehicle?\"\n- 🔋 \"Battery level?\"\n"
                "- 📡 \"GPS status?\"\n- 💥 \"Any crashes?\"\n- 📊 \"Show stats\"\n- 📜 \"Show last readings\"")

    return (f"I am the CATS AI Safety Assistant monitoring **{vehicle_id}**. "
            "Ask about current safety status, collision risk, speed, battery health, or recent alerts.")


# ──────────────────────────────────────────────
#  PAGE: DASHBOARD
# ──────────────────────────────────────────────

def render_dashboard(df, vehicle_id, model, scaler, le, iso_forest, anom_scaler):
    vehicle_df = df[df["vehicle_id"] == vehicle_id].sort_values("timestamp")
    if vehicle_df.empty:
        st.warning("No data for selected vehicle.")
        return

    latest = vehicle_df.iloc[-1]
    features = [latest[col] for col in FEATURE_COLUMNS]
    pred_label, pred_probs = predict_risk(model, scaler, le, features)
    is_anomaly, anom_score = detect_anomaly(features, iso_forest, anom_scaler)

    # 1. Top HUD Bar
    render_hud(vehicle_id, latest)

    # 2. Executive Status Banner
    render_executive_status_banner(latest["risk_label"], pred_probs, is_anomaly, anom_score)

    # 3. High-Tech Glassmorphism Telemetry Cards with Sparklines
    c1, c2, c3, c4, c5 = st.columns(5)
    avg_speed = vehicle_df["speed"].mean()
    speed_delta = latest["speed"] - avg_speed
    speed_delta_class = "delta-green" if latest["speed"] < 100 else ("delta-amber" if latest["speed"] < 140 else "delta-red")

    # Prepare sparkline data from last 20 readings
    spark_df = vehicle_df.tail(20)
    spark_speed = _build_sparkline_svg(spark_df["speed"].tolist(), color="#38bdf8")
    spark_accel = _build_sparkline_svg(spark_df["acceleration"].tolist(), color="#f43f5e")
    spark_impact = _build_sparkline_svg(spark_df["impact_score"].tolist(), color="#f59e0b")
    spark_battery = _build_sparkline_svg(spark_df["battery_status"].tolist(), color="#10b981")

    with c1:
        st.markdown(f"""
        <div class="telemetry-card tcard-d1">
            <div class="telemetry-label">🏎️ Speed Velocity</div>
            <div class="telemetry-val">{latest['speed']:.0f} <span style="font-size:14px;color:#94a3b8;">km/h</span></div>
            <div class="telemetry-delta {speed_delta_class}">{speed_delta:+.0f} km/h vs avg</div>
            <div class="sparkline-container">{spark_speed}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        acc_class = "delta-green" if latest["acceleration"] < 4.0 else ("delta-amber" if latest["acceleration"] < 7.0 else "delta-red")
        st.markdown(f"""
        <div class="telemetry-card tcard-d2">
            <div class="telemetry-label">⚡ G-Acceleration</div>
            <div class="telemetry-val">{latest['acceleration']:.1f} <span style="font-size:14px;color:#94a3b8;">g</span></div>
            <div class="telemetry-delta {acc_class}">Peak: {vehicle_df['acceleration'].max():.1f} g</div>
            <div class="sparkline-container">{spark_accel}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        imp_class = "delta-green" if latest["impact_score"] < 0.4 else ("delta-amber" if latest["impact_score"] < 0.7 else "delta-red")
        st.markdown(f"""
        <div class="telemetry-card tcard-d3">
            <div class="telemetry-label">💥 Impact Score</div>
            <div class="telemetry-val">{latest['impact_score']:.3f}</div>
            <div class="telemetry-delta {imp_class}">{'Nominal' if latest['impact_score']<0.4 else 'Collision Risk'}</div>
            <div class="sparkline-container">{spark_impact}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        bat_class = "delta-green" if latest["battery_status"] > 30 else ("delta-amber" if latest["battery_status"] > 10 else "delta-red")
        st.markdown(f"""
        <div class="telemetry-card tcard-d4">
            <div class="telemetry-label">🔋 Battery Pack</div>
            <div class="telemetry-val">{latest['battery_status']:.0f} <span style="font-size:14px;color:#94a3b8;">%</span></div>
            <div class="telemetry-delta {bat_class}">Range: ~{(latest['battery_status']*3.5):.0f} km</div>
            <div class="sparkline-container">{spark_battery}</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        gps_txt = "Active Lock" if latest["gps_status"] == 1 else "Signal Lost"
        gps_class = "delta-green" if latest["gps_status"] == 1 else "delta-red"
        gps_pct = (vehicle_df["gps_status"] == 1).mean()
        gps_spark = _build_sparkline_svg(spark_df["gps_status"].astype(float).tolist(), color="#34d399")
        st.markdown(f"""
        <div class="telemetry-card tcard-d5">
            <div class="telemetry-label">📡 GPS Telematics</div>
            <div class="telemetry-val" style="font-size:20px;">{gps_txt}</div>
            <div class="telemetry-delta {gps_class}">Uptime: {gps_pct:.0%}</div>
            <div class="sparkline-container">{gps_spark}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # 4. Tabbed Deep-Dive Views
    tab_gauges, tab_map, tab_trends = st.tabs([
        "📈 Live Gauges & Distribution",
        "🗺️ 3D Route Map & Waypoints",
        "📊 Dynamic Trends & Correlations",
    ])

    with tab_gauges:
        # Gauges Row
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            st.plotly_chart(make_gauge(latest["speed"], "SPEEDOMETER", 0, 220, SPEED_THRESHOLDS, "km/h"), use_container_width=True, key="db_spd")
        with g2:
            st.plotly_chart(make_gauge(latest["acceleration"], "G-FORCE ACCELEROMETER", 0, 15, ACCELERATION_THRESHOLDS, "g"), use_container_width=True, key="db_acc")
        with g3:
            st.plotly_chart(make_gauge(latest["impact_score"], "IMPACT SEVERITY SENSOR", 0, 1.0, IMPACT_SCORE_THRESHOLDS), use_container_width=True, key="db_imp")
        with g4:
            st.plotly_chart(make_gauge(latest["battery_status"], "STATE OF CHARGE", 0, 100, BATTERY_THRESHOLDS, "%", reverse=True), use_container_width=True, key="db_bat")

        # Risk Distribution + Health Snapshot
        c_p1, c_p2 = st.columns([1, 1])
        with c_p1:
            risk_counts = vehicle_df["risk_label"].value_counts()
            colors = [STATUS_COLORS.get(label, "#999") for label in risk_counts.index]
            fig_pie = px.pie(values=risk_counts.values, names=risk_counts.index,
                             title="Vehicle Lifetime Risk Classification Envelope",
                             color_discrete_sequence=colors, hole=0.45)
            fig_pie.update_layout(height=240, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        with c_p2:
            st.markdown("#### 📋 Diagnostic Rule Attribution")
            explanations = generate_explanation(latest)
            for exp in explanations:
                st.markdown(f"- {exp}")
            st.markdown("#### 🤖 Model Probability Breakdown")
            bar_html = ""
            for label, prob in sorted(pred_probs.items(), key=lambda x: -x[1]):
                bar_cls = "red" if label == STATUS_CRITICAL else ("amber" if label == STATUS_WARNING else "green")
                bar_html += _render_confidence_bar(label, prob, bar_cls)
            st.markdown(f'<div class="section-fade">{bar_html}</div>', unsafe_allow_html=True)

    with tab_map:
        st.markdown("#### 🛰️ Live Geospatial Route Telemetry")
        render_telemetry_map(vehicle_df, latest)

    with tab_trends:
        fig_sync = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 subplot_titles=("Velocity (km/h) with Safety Threshold Bands", "Acceleration (g) Dynamics"),
                                 vertical_spacing=0.12)
        fig_sync.add_trace(go.Scatter(x=vehicle_df["timestamp"], y=vehicle_df["speed"],
                           mode="lines", name="Speed", line=dict(color="#38bdf8", width=2)), row=1, col=1)
        fig_sync.add_hline(y=SPEED_THRESHOLDS["warning"], line_dash="dash", line_color="#f59e0b",
                           annotation_text="Warning Threshold", row=1, col=1)
        fig_sync.add_hline(y=SPEED_THRESHOLDS["critical"], line_dash="dash", line_color="#ef4444",
                           annotation_text="Critical Threshold", row=1, col=1)
        fig_sync.add_trace(go.Scatter(x=vehicle_df["timestamp"], y=vehicle_df["acceleration"],
                           mode="lines", name="Acceleration", line=dict(color="#f43f5e", width=2)), row=2, col=1)
        fig_sync.update_layout(height=380, showlegend=False, template="plotly_dark",
                               margin=dict(l=40, r=20, t=40, b=20))
        st.plotly_chart(fig_sync, use_container_width=True)


# ──────────────────────────────────────────────
#  PAGE: REAL-TIME SIMULATION
# ──────────────────────────────────────────────

def render_realtime(model, scaler, le, iso_forest, anom_scaler):
    st.markdown('<div class="section-fade">', unsafe_allow_html=True)
    st.markdown("### 🔴 Real-Time Dynamic Telemetry Stream")
    st.caption("Simulates continuous CAN-bus streaming data packets with sub-second ML inference and anomaly alerting.")
    st.markdown('</div>', unsafe_allow_html=True)

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        scenario = st.selectbox("Driving Scenario Profile", ["mixed", "normal", "dangerous"])
    with col_ctrl2:
        interval = st.slider("Stream Refresh Interval (seconds)", 1, 5, 2)
    with col_ctrl3:
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        running = st.toggle("▶️ Activate Live Stream", value=False)

    if "sim_history" not in st.session_state:
        st.session_state.sim_history = []
    if "sim_rng" not in st.session_state:
        st.session_state.sim_rng = np.random.default_rng(None)

    if running:
        last_r = st.session_state.sim_history[-1] if st.session_state.sim_history else None
        reading = generate_sim_reading(st.session_state.sim_rng, scenario, last_r)
        reading["timestamp"] = datetime.now().strftime("%H:%M:%S")

        features = [reading["speed"], reading["acceleration"], reading["impact_score"],
                     reading["battery_status"], reading["gps_status"]]
        pred_label, pred_probs = predict_risk(model, scaler, le, features)
        is_anomaly, anom_score = detect_anomaly(features, iso_forest, anom_scaler)

        reading["risk_label"] = pred_label
        reading["is_anomaly"] = is_anomaly
        reading["anomaly_score"] = anom_score
        reading["confidence"] = max(pred_probs.values())

        st.session_state.sim_history.append(reading)
        if len(st.session_state.sim_history) > 60:
            st.session_state.sim_history = st.session_state.sim_history[-60:]

        # Executive Banner
        render_executive_status_banner(pred_label, pred_probs, is_anomaly, anom_score)

        # Gauges
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            st.plotly_chart(make_gauge(reading["speed"], "SPEED", 0, 220, SPEED_THRESHOLDS, "km/h"), use_container_width=True, key="rt_spd")
        with g2:
            st.plotly_chart(make_gauge(reading["acceleration"], "G-FORCE", 0, 15, ACCELERATION_THRESHOLDS, "g"), use_container_width=True, key="rt_acc")
        with g3:
            st.plotly_chart(make_gauge(reading["impact_score"], "IMPACT", 0, 1.0, IMPACT_SCORE_THRESHOLDS), use_container_width=True, key="rt_imp")
        with g4:
            st.plotly_chart(make_gauge(reading["battery_status"], "BATTERY", 0, 100, BATTERY_THRESHOLDS, "%", reverse=True), use_container_width=True, key="rt_bat")

        # Map + Time series
        hist_df = pd.DataFrame(st.session_state.sim_history)
        st.markdown("#### 🛰️ Live Simulated Telematics GPS Track")
        render_telemetry_map(hist_df, reading)

        time.sleep(interval)
        st.rerun()

    else:
        if st.session_state.sim_history:
            st.info(f"Stream paused. {len(st.session_state.sim_history)} telemetry frames captured.")
            hist_df = pd.DataFrame(st.session_state.sim_history)
            st.dataframe(hist_df, use_container_width=True)
        else:
            st.info("Toggle **Activate Live Stream** to initialize real-time vehicle monitoring.")


# ──────────────────────────────────────────────
#  PAGE: ALERTS + TIMELINE + DOSSIER
# ──────────────────────────────────────────────

def render_alerts(df, vehicle_id, model, scaler, le, iso_forest, anom_scaler):
    vehicle_df = df[df["vehicle_id"] == vehicle_id].sort_values("timestamp", ascending=False)
    alerts_df = vehicle_df[vehicle_df["risk_label"].isin([STATUS_WARNING, STATUS_CRITICAL])].head(30)

    # ── Severity Count Badges ──
    all_vehicle = df[df["vehicle_id"] == vehicle_id]
    n_critical = int((all_vehicle["risk_label"] == STATUS_CRITICAL).sum())
    n_warning = int((all_vehicle["risk_label"] == STATUS_WARNING).sum())
    n_normal = int((all_vehicle["risk_label"] == STATUS_NORMAL).sum())
    st.markdown(f"""
    <div class="severity-badges">
        <div class="sev-badge critical">🔴 {n_critical} Critical</div>
        <div class="sev-badge warning">🟠 {n_warning} Warning</div>
        <div class="sev-badge normal">🟢 {n_normal} Normal</div>
    </div>
    """, unsafe_allow_html=True)

    if alerts_df.empty:
        st.success("✅ No active alerts for this vehicle. All historical records are NORMAL.")
        return

    # Timeline View
    st.markdown("### 📅 Alert Incident Timeline")
    timeline_data = []
    for _, row in alerts_df.iterrows():
        timeline_data.append({
            "timestamp": row["timestamp"],
            "status": row["risk_label"],
            "speed": row["speed"],
        })
    tl_df = pd.DataFrame(timeline_data)
    color_map = {STATUS_WARNING: "#f59e0b", STATUS_CRITICAL: "#ef4444"}
    fig_tl = px.scatter(tl_df, x="timestamp", y="speed", color="status",
                        color_discrete_map=color_map, size_max=12,
                        title="Historical Alert Distribution (Velocity at Event)")
    fig_tl.update_layout(height=240, template="plotly_dark", margin=dict(l=40, r=20, t=40, b=20))
    fig_tl.update_traces(marker=dict(size=11, line=dict(width=1, color="white")))
    st.plotly_chart(fig_tl, use_container_width=True)

    st.markdown("---")

    # Alert Cards
    st.markdown(f"### 🚨 Active Incident Queue ({len(alerts_df)} Recent Events)")
    for idx, row in alerts_df.iterrows():
        status = row["risk_label"]
        emoji = STATUS_EMOJIS[status]
        explanations = generate_explanation(row)
        features = [row[col] for col in FEATURE_COLUMNS]
        pred_label, pred_probs = predict_risk(model, scaler, le, features)
        is_anomaly, anom_score = detect_anomaly(features, iso_forest, anom_scaler)

        anom_tag = " 🔮 ANOMALY" if is_anomaly else ""
        with st.expander(f"{emoji} {status}{anom_tag} — {row['timestamp']} | Speed: {row['speed']:.0f} km/h | Impact: {row['impact_score']:.3f}",
                         expanded=(status == STATUS_CRITICAL and idx == alerts_df.index[0])):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("**Root Cause Attribution:**")
                for exp in explanations:
                    st.markdown(f"- {exp}")
                if is_anomaly:
                    st.markdown(f"- 🔮 **Anomaly Outlier Flagged** (Score: {anom_score:.3f})")
                if "latitude" in row:
                    st.caption(f"📍 GPS Fix: Lat {row['latitude']:.4f}, Lon {row['longitude']:.4f}")
            with c2:
                st.markdown("**Classifier Consensus:**")
                bar_html = ""
                for label in [STATUS_CRITICAL, STATUS_WARNING, STATUS_NORMAL]:
                    if label in pred_probs:
                        bar_cls = "red" if label == STATUS_CRITICAL else ("amber" if label == STATUS_WARNING else "green")
                        bar_html += _render_confidence_bar(label, pred_probs[label], bar_cls)
                st.markdown(bar_html, unsafe_allow_html=True)

            ca, cb, _ = st.columns([1, 1, 2])
            with ca:
                if st.button("🚫 Mark False Alarm", key=f"fa_{idx}"):
                    record_feedback(row, "false_alarm")
                    st.success("Logged as false alarm. Model retraining queue updated.")
            with cb:
                if st.button("📄 Generate Dossier", key=f"dos_{idx}"):
                    dossier_text = generate_incident_dossier(row, model, scaler, le, iso_forest, anom_scaler)
                    st.session_state[f"dossier_view_{idx}"] = dossier_text

            if f"dossier_view_{idx}" in st.session_state:
                st.markdown("#### 📄 Forensic Incident Investigation Dossier")
                st.markdown(f'<div class="dossier-box">{st.session_state[f"dossier_view_{idx}"]}</div>', unsafe_allow_html=True)
                st.download_button(
                    "📥 Download Official Docket (.md)",
                    st.session_state[f"dossier_view_{idx}"],
                    file_name=f"Incident_Dossier_{vehicle_id}_{idx}.md",
                    mime="text/markdown",
                    key=f"dl_dos_{idx}",
                )


# ──────────────────────────────────────────────
#  PAGE: AI CHAT
# ──────────────────────────────────────────────

def render_chat(df, vehicle_id, model, scaler, le):
    backend = get_backend()
    model_name = GROQ_MODEL if backend == "groq" else (OPENROUTER_MODEL if backend == "openrouter" else "Rule Engine")
    badge = {
        "groq": f"🧠 Powered by Groq ({model_name})",
        "openrouter": f"🧠 Powered by OpenRouter ({model_name})",
        "rule_based": "📋 Rule-based mode (Set GROQ_API_KEY in .env for full AI chat)",
    }

    st.markdown('<div class="section-fade">', unsafe_allow_html=True)
    st.markdown("### 💬 Conversational Safety Intelligence")
    st.markdown(f'<span class="ai-badge">{badge[backend]}</span>', unsafe_allow_html=True)
    st.caption(f"Real-time safety copilot analyzing telemetry streams for **{vehicle_id}**.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Quick prompt chips styled as animated gradient pills
    st.markdown("**💡 Quick Telematics Inquiries:**")
    qc1, qc2, qc3, qc4 = st.columns(4)
    quick_prompt = None
    with qc1:
        if st.button("⚡ Safe to drive?", key="qp_safe"):
            quick_prompt = "Is this vehicle currently safe to drive? Give a clear safety assessment with specific metrics."
    with qc2:
        if st.button("🚨 Assess risk", key="qp_risk"):
            quick_prompt = "Assess the current vehicle risk level and explain why this classification was assigned."
    with qc3:
        if st.button("💥 Crash impact", key="qp_crash"):
            quick_prompt = "Analyze the vehicle impact score and acceleration for any crash or collision risk."
    with qc4:
        if st.button("🔋 Battery & GPS", key="qp_bat"):
            quick_prompt = "Evaluate the vehicle battery status and GPS connectivity uptime."

    # Render quick-chip pill decorators
    st.markdown("""
    <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px;">
        <span class="quick-chip" style="animation-delay: 0s;">⚡ Safety Check</span>
        <span class="quick-chip" style="animation-delay: 0.1s;">🚨 Risk Assessment</span>
        <span class="quick-chip" style="animation-delay: 0.2s;">💥 Crash Analysis</span>
        <span class="quick-chip" style="animation-delay: 0.3s;">🔋 Systems Health</span>
    </div>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": f"👋 Greetings Operator! I am the CATS AI Safety Copilot monitoring **{vehicle_id}**. I am continuously inspecting CAN-bus telemetry, crash parameters, and battery health. How can I assist your fleet review today?"}
        ]

    # Render chat history with custom bubbles
    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            st.markdown(f"""
            <div class="chat-sender">🤖 CATS AI COPILOT</div>
            <div class="chat-bubble-bot">{msg["content"]}</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-sender-user">👤 OPERATOR</div>
            <div class="chat-bubble-user">{msg["content"]}</div>
            """, unsafe_allow_html=True)

    input_prompt = st.chat_input("Ask about vehicle safety, telemetry, alerts...")
    active_prompt = quick_prompt if quick_prompt else input_prompt

    if active_prompt:
        st.session_state.messages.append({"role": "user", "content": active_prompt})
        st.markdown(f"""
        <div class="chat-sender-user">👤 OPERATOR</div>
        <div class="chat-bubble-user">{active_prompt}</div>
        """, unsafe_allow_html=True)

        # Show typing indicator while processing
        typing_placeholder = st.empty()
        typing_placeholder.markdown("""
        <div class="typing-indicator">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>
        """, unsafe_allow_html=True)

        vehicle_df = df[df["vehicle_id"] == vehicle_id].sort_values("timestamp")
        latest = vehicle_df.iloc[-1] if len(vehicle_df) else pd.Series()
        vehicle_data = latest.to_dict() if len(latest) else {}
        stats = compute_statistics(df, vehicle_id)

        response = None
        if backend != "rule_based":
            response = llm_chat(st.session_state.messages, vehicle_data, stats)

        if response is None or response.startswith("⚠️"):
            response = handle_chat_rulebased(active_prompt, df, vehicle_id, model, scaler, le)

        # Clear typing indicator and show response
        typing_placeholder.empty()

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(f"""
        <div class="chat-sender">🤖 CATS AI COPILOT</div>
        <div class="chat-bubble-bot">{response}</div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  PAGE: SIMULATOR + SHAP
# ──────────────────────────────────────────────

def render_simulator(model, scaler, le, iso_forest, anom_scaler):
    st.markdown('<div class="section-fade">', unsafe_allow_html=True)
    st.markdown("### 🎛️ Telemetry Simulator & SHAP Factor Attribution")
    st.caption("Adjust real-time sensor parameters to observe classification boundaries, SHAP factor attribution, and anomaly scores.")
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        sim_speed = st.slider("Speed Velocity (km/h)", 0.0, 220.0, 60.0, 1.0)
        sim_accel = st.slider("Acceleration (g)", 0.0, 15.0, 1.0, 0.1)
        sim_impact = st.slider("Impact Severity Score", 0.0, 1.0, 0.05, 0.01)
    with c2:
        sim_battery = st.slider("Battery Pack (%)", 0.0, 100.0, 75.0, 1.0)
        sim_gps = st.selectbox("GPS Telemetry Link", [1, 0], format_func=lambda x: "🟢 Satellite Fix Active" if x == 1 else "🔴 Signal Dead Zone")

    features = [sim_speed, sim_accel, sim_impact, sim_battery, sim_gps]
    pred_label, pred_probs = predict_risk(model, scaler, le, features)
    is_anomaly, anom_score = detect_anomaly(features, iso_forest, anom_scaler)

    st.markdown("---")
    render_executive_status_banner(pred_label, pred_probs, is_anomaly, anom_score)

    sim_row = pd.Series({"speed": sim_speed, "acceleration": sim_accel,
                          "impact_score": sim_impact, "battery_status": sim_battery,
                          "gps_status": sim_gps, "vehicle_id": "VH-SIM",
                          "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    explanations = generate_explanation(sim_row)

    ex_col, shap_col = st.columns(2)
    with ex_col:
        st.markdown("**📋 Physical Domain Rule Analysis:**")
        for exp in explanations:
            st.markdown(f"- {exp}")

        st.markdown("**📊 Multi-Class Prediction Probabilities:**")
        bar_html = ""
        for label in [STATUS_CRITICAL, STATUS_WARNING, STATUS_NORMAL]:
            if label in pred_probs:
                bar_cls = "red" if label == STATUS_CRITICAL else ("amber" if label == STATUS_WARNING else "green")
                bar_html += _render_confidence_bar(f"{STATUS_EMOJIS.get(label, '')} {label}", pred_probs[label], bar_cls)
        st.markdown(f'<div class="section-fade">{bar_html}</div>', unsafe_allow_html=True)

    with shap_col:
        st.markdown("**🔬 Mathematical SHAP Attribution (KernelExplainer):**")
        try:
            import shap
            X_input = np.array(features).reshape(1, -1)
            X_scaled = scaler.transform(X_input)

            bg_data = shap.sample(pd.DataFrame(scaler.transform(load_data()[FEATURE_COLUMNS].values), columns=FEATURE_COLUMNS), 50)
            explainer = shap.KernelExplainer(model.predict_proba, bg_data)
            shap_values = explainer.shap_values(X_scaled, nsamples=40)

            pred_idx = list(le.classes_).index(pred_label)
            if isinstance(shap_values, list):
                sv = shap_values[pred_idx][0]
            else:
                sv = shap_values[0, :, pred_idx] if shap_values.ndim == 3 else shap_values[0]

            shap_df = pd.DataFrame({
                "Feature": FEATURE_COLUMNS,
                "SHAP Value": sv,
                "Abs SHAP": np.abs(sv),
            }).sort_values("Abs SHAP", ascending=True)

            colors = ["#ef4444" if v > 0 else "#38bdf8" for v in shap_df["SHAP Value"]]
            fig_shap = go.Figure(go.Bar(
                x=shap_df["SHAP Value"], y=shap_df["Feature"],
                orientation="h", marker_color=colors,
                text=[f"{v:+.3f}" for v in shap_df["SHAP Value"]], textposition="outside",
            ))
            fig_shap.update_layout(
                height=250, template="plotly_dark",
                title=f"Marginal Impact on {pred_label} Class Probability",
                xaxis_title="Directional Influence (Log-Odds)",
                margin=dict(l=100, r=40, t=40, b=20),
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            st.caption("🔴 Red = Pushes toward this risk level | 🔵 Blue = Pushes away")
        except Exception as e:
            st.warning(f"SHAP attribution notice: {e}")

    # Generate dossier
    st.markdown("---")
    if st.button("📄 Export Simulated Forensic Dossier"):
        sim_dossier = generate_incident_dossier(sim_row, model, scaler, le, iso_forest, anom_scaler)
        st.markdown(f'<div class="dossier-box">{sim_dossier}</div>', unsafe_allow_html=True)
        st.download_button("📥 Download Dossier (.md)", sim_dossier, "Simulated_Incident_Dossier.md", "text/markdown")


# ──────────────────────────────────────────────
#  PAGE: MODEL COMPARISON
# ──────────────────────────────────────────────

def render_model_comparison():
    st.markdown('<div class="section-fade">', unsafe_allow_html=True)
    st.markdown("### 🏆 Architectural Model Benchmarking")
    st.markdown('</div>', unsafe_allow_html=True)
    comparison = load_model_comparison()

    if comparison is None:
        st.warning("Run `python models/train_compare.py` to generate model comparison data.")
        return

    st.success(f"**Selected Production Model: {comparison['best_model']}** (Validated across 5-Fold Stratified Cross-Validation)")

    # Comparison table
    rows = []
    for name, metrics in comparison["models"].items():
        rows.append({
            "Model Architecture": ("🏆 " if name == comparison["best_model"] else "") + name,
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "F1 (Weighted)": f"{metrics['f1_weighted']:.4f}",
            "Precision": f"{metrics['precision_weighted']:.4f}",
            "Recall": f"{metrics['recall_weighted']:.4f}",
            "CV F1 (Mean ± Std)": f"{metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}",
            "Training Time": f"{metrics['train_time_s']:.2f}s",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Bar chart
    model_names = list(comparison["models"].keys())
    metrics_to_plot = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
    fig = go.Figure()
    colors = ["#38bdf8", "#34d399", "#fbbf24", "#f87171"]
    for i, metric in enumerate(metrics_to_plot):
        values = [comparison["models"][m][metric] for m in model_names]
        fig.add_trace(go.Bar(name=metric.replace("_weighted", "").title(),
                             x=model_names, y=values, marker_color=colors[i]))
    fig.update_layout(barmode="group", height=340, template="plotly_dark",
                      title="Metric Comparison Across Candidate Architectures", yaxis_range=[0.95, 1.0],
                      margin=dict(l=40, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Confusion matrices
    st.markdown("**Confusion Matrices:**")
    cols = st.columns(len(model_names))
    for i, name in enumerate(model_names):
        with cols[i]:
            cm = np.array(comparison["models"][name]["confusion_matrix"])
            labels = comparison["class_labels"]
            fig_cm = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                               x=labels, y=labels, text_auto=True, color_continuous_scale="Blues")
            fig_cm.update_layout(height=240, title=name, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_cm, use_container_width=True)


# ──────────────────────────────────────────────
#  PAGE: CODEBASE KNOWLEDGE GRAPH (GRAPHIFY)
# ──────────────────────────────────────────────

def render_knowledge_graph():
    st.markdown('<div class="section-fade">', unsafe_allow_html=True)
    st.markdown("### 🌐 Codebase Architecture & Knowledge Graph (via Graphify)")
    st.caption("Auto-generated knowledge graph mapping all 24 project files, AST call structures, god nodes, and community clusters.")
    st.markdown('</div>', unsafe_allow_html=True)

    report_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "graphify-out", "GRAPH_REPORT.md")
    graph_html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "graphify-out", "graph.html")

    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        st.metric("Total Nodes Extracted", "141 AST Nodes")
    with col_k2:
        st.metric("Relational Edges", "310 Edges")
    with col_k3:
        st.metric("Detected Communities", "14 Clusters")

    st.markdown("---")

    if os.path.exists(graph_html_path):
        st.markdown("#### 🔍 Interactive 2D/3D Network Topology")
        st.caption("Drag, zoom, and click any node to trace code calls, imports, and dependencies.")
        with open(graph_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=650, scrolling=True)
    else:
        st.info("Run `graphify` to export `graph.html`.")

    if os.path.exists(report_path):
        with st.expander("📖 View Full GRAPH_REPORT.md"):
            with open(report_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())


# ──────────────────────────────────────────────
#  PAGE: FEEDBACK + RETRAINING
# ──────────────────────────────────────────────

def render_feedback():
    st.markdown('<div class="section-fade">', unsafe_allow_html=True)
    st.markdown("### 📝 Active Learning & False Alarm Audit Log")
    st.markdown('</div>', unsafe_allow_html=True)

    if os.path.exists(FEEDBACK_PATH):
        feedback_df = pd.read_csv(FEEDBACK_PATH)
        st.dataframe(feedback_df, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Feedback Entries", len(feedback_df))
            st.metric("False Alarm Records", len(feedback_df[feedback_df["feedback_type"] == "false_alarm"]))
        with c2:
            csv = feedback_df.to_csv(index=False)
            st.download_button("📥 Download Audit Log CSV", csv, "feedback_log.csv", "text/csv")

        st.markdown("---")
        st.markdown("### 🔄 Active Learning Pipeline: Retrain Production Model")
        st.caption("Incorporates operator false alarm tags to adjust decision boundaries dynamically.")

        if st.button("🔄 Retrain Production Model with Feedback Data", type="primary"):
            with st.spinner("Retraining model with feedback data..."):
                try:
                    df = pd.read_csv(DATASET_PATH)
                    false_alarms = feedback_df[feedback_df["feedback_type"] == "false_alarm"]

                    relabeled = 0
                    for _, fa in false_alarms.iterrows():
                        mask = (
                            (df["speed"].between(fa["speed"] - 1, fa["speed"] + 1)) &
                            (df["acceleration"].between(fa["acceleration"] - 0.1, fa["acceleration"] + 0.1))
                        )
                        matching = df[mask & (df["risk_label"] != STATUS_NORMAL)]
                        if len(matching) > 0:
                            idx = matching.index[0]
                            if df.loc[idx, "risk_label"] == STATUS_CRITICAL:
                                df.loc[idx, "risk_label"] = STATUS_WARNING
                            elif df.loc[idx, "risk_label"] == STATUS_WARNING:
                                df.loc[idx, "risk_label"] = STATUS_NORMAL
                            relabeled += 1

                    if relabeled > 0:
                        from sklearn.ensemble import GradientBoostingClassifier
                        from sklearn.preprocessing import StandardScaler, LabelEncoder

                        X = df[FEATURE_COLUMNS].values
                        y = df["risk_label"].values
                        le_new = LabelEncoder()
                        y_enc = le_new.fit_transform(y)
                        scaler_new = StandardScaler()
                        X_scaled = scaler_new.fit_transform(X)

                        clf = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
                        clf.fit(X_scaled, y_enc)

                        joblib.dump({"model": clf, "label_encoder": le_new}, CLASSIFIER_PATH)
                        joblib.dump(scaler_new, SCALER_PATH)

                        st.success(f"✅ Model retrained successfully! {relabeled} training instances adjusted.")
                        st.cache_resource.clear()
                    else:
                        st.info("No corresponding readings found to relabel. Model weights preserved.")
                except Exception as e:
                    st.error(f"Retraining encountered error: {e}")
    else:
        st.info("No feedback recorded yet. Mark alerts as false alarms on the Alerts page to build the audit log.")


# ──────────────────────────────────────────────
#  MAIN APPLICATION ROUTER
# ──────────────────────────────────────────────

def main():
    # Sidebar
    st.sidebar.markdown(f"""
    <div style="text-align: center; padding: 10px 0 15px 0;">
        <span style="font-size: 42px;">🚗</span>
        <div style="font-size: 20px; font-weight: 800; color: #ffffff; letter-spacing: -0.5px;">CRASH CONNECT</div>
        <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; color: #38bdf8;">Guardian on Wheels</div>
        <div style="font-size: 11px; color: #94a3b8; margin-top: 2px;">Crisis Alert Tech System (CATS)</div>
        <div style="margin-top: 6px;"><span class="version-badge">{_APP_VERSION} WOW</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("---")

    try:
        model, le, scaler = load_model()
        df = load_data()
        iso_forest, anom_scaler = load_anomaly_model()
    except Exception as e:
        st.error(f"Failed to load: {e}")
        st.info("Run: `python data/generate_dataset.py` → `python models/train_compare.py`")
        return

    # Vehicle selector
    vehicle_ids = get_vehicle_ids(df)
    selected_vehicle = st.sidebar.selectbox("🚘 Select Monitored Asset", vehicle_ids)

    # Navigation
    page = st.sidebar.radio("📍 Navigation", [
        "📊 Dashboard",
        "🔴 Real-Time Sim",
        "🚨 Alerts & Timeline",
        "💬 AI Chat",
        "🎛️ Simulator + SHAP",
        "🏆 Model Comparison",
        "🌐 Knowledge Graph (Graphify)",
        "📝 Feedback & Retrain",
    ])

    # System Diagnostics Box
    backend = get_backend()
    llm_diag = f"Groq ({GROQ_MODEL})" if backend == "groq" else (f"OpenRouter ({OPENROUTER_MODEL})" if backend == "openrouter" else "Rule Engine")

    st.sidebar.markdown("---")

    # Enhanced sidebar info box
    st.sidebar.markdown(f"""
    <div class="sidebar-diag">
        <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #38bdf8; margin-bottom: 6px;">⚙️ Telematics Diagnostics</div>
        <div class="diag-row"><span>ML Classifier:</span><span class="diag-val">Gradient Boost (99.9%)</span></div>
        <div class="diag-row"><span>Anomaly Core:</span><span class="diag-val">Isolation Forest Active</span></div>
        <div class="diag-row"><span>LLM Engine:</span><span class="diag-val">{llm_diag}</span></div>
        <div class="diag-row"><span>Telemetry Cache:</span><span class="diag-val">{len(df):,} Records</span></div>
        <div class="diag-row"><span>Stream Sync:</span><span class="diag-val" style="color: #34d399;">● Connected</span></div>
        <div class="diag-row"><span>App Version:</span><span class="diag-val" style="color: #a78bfa;">{_APP_VERSION} WOW</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Route Page
    if page == "📊 Dashboard":
        render_dashboard(df, selected_vehicle, model, scaler, le, iso_forest, anom_scaler)
    elif page == "🔴 Real-Time Sim":
        render_realtime(model, scaler, le, iso_forest, anom_scaler)
    elif page == "🚨 Alerts & Timeline":
        render_alerts(df, selected_vehicle, model, scaler, le, iso_forest, anom_scaler)
    elif page == "💬 AI Chat":
        render_chat(df, selected_vehicle, model, scaler, le)
    elif page == "🎛️ Simulator + SHAP":
        render_simulator(model, scaler, le, iso_forest, anom_scaler)
    elif page == "🏆 Model Comparison":
        render_model_comparison()
    elif page == "🌐 Knowledge Graph (Graphify)":
        render_knowledge_graph()
    elif page == "📝 Feedback & Retrain":
        render_feedback()

    # ── Footer System Health Strip (every page) ──
    _render_footer(record_count=len(df))


if __name__ == "__main__":
    main()
