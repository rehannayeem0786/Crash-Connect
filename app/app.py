"""
CATS AI Safety Bot — Main Streamlit Application (v2.0)

Upgraded with:
- Groq/OpenRouter AI-powered conversational chat
- Real-time telemetry simulation
- SHAP per-prediction explanations
- Multi-model comparison display
- Isolation Forest anomaly detection
- Alert timeline visualization
- Feedback-driven model retraining
- Polished UI with animated gauges & maps
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os
import sys
import time
from datetime import datetime, timedelta

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
from utils.llm import chat as llm_chat, get_backend

# === Page Config ===
st.set_page_config(
    page_title="CATS AI Safety Bot",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === Custom CSS ===
st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; }
    .status-normal {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        color: white; padding: 20px; border-radius: 12px;
        text-align: center; font-size: 24px; font-weight: bold;
        box-shadow: 0 4px 15px rgba(46,204,113,0.3);
    }
    .status-warning {
        background: linear-gradient(135deg, #f39c12, #e67e22);
        color: white; padding: 20px; border-radius: 12px;
        text-align: center; font-size: 24px; font-weight: bold;
        box-shadow: 0 4px 15px rgba(243,156,18,0.3);
    }
    .status-critical {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white; padding: 20px; border-radius: 12px;
        text-align: center; font-size: 24px; font-weight: bold;
        box-shadow: 0 4px 15px rgba(231,76,60,0.3);
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 4px 15px rgba(231,76,60,0.3); }
        50% { box-shadow: 0 4px 25px rgba(231,76,60,0.6); }
        100% { box-shadow: 0 4px 15px rgba(231,76,60,0.3); }
    }
    .anomaly-badge {
        background: #9b59b6; color: white; padding: 4px 12px;
        border-radius: 20px; font-size: 12px; font-weight: bold;
        display: inline-block; margin-left: 8px;
    }
    .ai-badge {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; padding: 3px 10px; border-radius: 12px;
        font-size: 11px; font-weight: bold; display: inline-block;
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
#  HELPER FUNCTIONS
# ──────────────────────────────────────────────

def predict_risk(model, scaler, le, features):
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    label = le.inverse_transform([prediction])[0]
    return label, dict(zip(le.classes_, probabilities))


def detect_anomaly(features, iso_forest, anom_scaler):
    if iso_forest is None:
        return False, 0.0
    X = np.array(features).reshape(1, -1)
    X_scaled = anom_scaler.transform(X)
    pred = iso_forest.predict(X_scaled)[0]
    score = iso_forest.decision_function(X_scaled)[0]
    return pred == -1, float(score)


def render_status_banner(status):
    css_class = f"status-{status.lower()}"
    emoji = STATUS_EMOJIS.get(status, "")
    st.markdown(f'<div class="{css_class}">{emoji} Vehicle Status: {status} {emoji}</div>', unsafe_allow_html=True)


def make_gauge(value, title, min_val, max_val, thresholds, unit="", reverse=False):
    """Create an animated gauge chart."""
    if reverse:
        steps = [
            {"range": [min_val, thresholds.get("critical", 10)], "color": "#e74c3c"},
            {"range": [thresholds.get("critical", 10), thresholds.get("warning", 30)], "color": "#f39c12"},
            {"range": [thresholds.get("warning", 30), max_val], "color": "#2ecc71"},
        ]
        thresh_val = thresholds.get("warning", 30)
    else:
        steps = [
            {"range": [min_val, thresholds.get("warning", 60)], "color": "#2ecc71"},
            {"range": [thresholds.get("warning", 60), thresholds.get("critical", 80)], "color": "#f39c12"},
            {"range": [thresholds.get("critical", 80), max_val], "color": "#e74c3c"},
        ]
        thresh_val = thresholds.get("critical", 80)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 14, "color": "white"}},
        number={"suffix": f" {unit}", "font": {"color": "white", "size": 22}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickcolor": "white", "tickfont": {"color": "white"}},
            "bar": {"color": "#3498db"},
            "steps": steps,
            "threshold": {"line": {"color": "white", "width": 2}, "thickness": 0.8, "value": thresh_val},
        },
    ))
    fig.update_layout(
        height=180, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
    )
    return fig


def generate_sim_reading(rng, scenario="mixed"):
    """Generate a single simulated telemetry reading."""
    if scenario == "normal":
        return {
            "speed": float(np.clip(rng.normal(60, 15), 0, 95)),
            "acceleration": float(np.clip(rng.exponential(0.8), 0, 3)),
            "impact_score": float(np.clip(rng.exponential(0.03), 0, 0.3)),
            "battery_status": float(np.clip(rng.normal(80, 10), 40, 100)),
            "gps_status": int(rng.choice([1]*19 + [0])),
        }
    elif scenario == "dangerous":
        return {
            "speed": float(np.clip(rng.normal(140, 25), 80, 220)),
            "acceleration": float(np.clip(rng.normal(6, 2), 2, 12)),
            "impact_score": float(np.clip(rng.normal(0.6, 0.2), 0.1, 1.0)),
            "battery_status": float(np.clip(rng.normal(20, 10), 0, 50)),
            "gps_status": int(rng.choice([1, 1, 0, 0, 0])),
        }
    else:  # mixed
        r = rng.random()
        if r < 0.6:
            return generate_sim_reading(rng, "normal")
        elif r < 0.85:
            return {
                "speed": float(np.clip(rng.normal(100, 20), 40, 145)),
                "acceleration": float(np.clip(rng.normal(3.5, 1.5), 0.5, 7)),
                "impact_score": float(np.clip(rng.normal(0.3, 0.15), 0, 0.7)),
                "battery_status": float(np.clip(rng.normal(45, 15), 10, 80)),
                "gps_status": int(rng.choice([1, 1, 1, 0])),
            }
        else:
            return generate_sim_reading(rng, "dangerous")


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
        resp = f"**Current Status: {STATUS_EMOJIS[status]} {status}**\n\n"
        resp += f"- Speed: {latest['speed']:.0f} km/h\n- Acceleration: {latest['acceleration']:.1f} g\n"
        resp += f"- Impact Score: {latest['impact_score']:.3f}\n- Battery: {latest['battery_status']:.0f}%\n"
        resp += f"- GPS: {'Active' if latest['gps_status'] == 1 else 'Lost'}\n\n**Analysis:**\n"
        for exp in explanations:
            resp += f"- {exp}\n"
        return resp

    if any(kw in q for kw in ["why", "reason", "explain", "cause", "triggered"]):
        explanations = generate_explanation(latest)
        status = latest["risk_label"]
        resp = f"The latest reading was classified as **{status}** because:\n\n"
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
                f"- {'⚠️ Above warning threshold!' if latest['speed'] >= 100 else '✅ Within normal range.'}")

    if any(kw in q for kw in ["battery", "charge", "power"]):
        return (f"**Battery for {vehicle_id}:**\n\n- Current: {latest['battery_status']:.0f}%\n"
                f"- Average: {vehicle_df['battery_status'].mean():.0f}%\n"
                f"- {'🔴 Critically low!' if latest['battery_status'] <= 10 else '⚠️ Low.' if latest['battery_status'] <= 30 else '✅ OK.'}")

    if any(kw in q for kw in ["gps", "location", "signal"]):
        gps_pct = (vehicle_df["gps_status"] == 1).mean() * 100
        return (f"**GPS for {vehicle_id}:**\n\n- Current: {'🟢 Active' if latest['gps_status'] == 1 else '🔴 Lost'}\n"
                f"- Uptime: {gps_pct:.1f}%\n- Losses: {(vehicle_df['gps_status'] == 0).sum()}")

    if any(kw in q for kw in ["impact", "crash", "collision"]):
        return (f"**Impact for {vehicle_id}:**\n\n- Current: {latest['impact_score']:.3f}\n"
                f"- Max: {vehicle_df['impact_score'].max():.3f}\n"
                f"- High events (>0.7): {(vehicle_df['impact_score'] > 0.7).sum()}")

    if any(kw in q for kw in ["history", "last", "recent", "readings", "show"]):
        recent = vehicle_df.tail(10)[["timestamp", "speed", "acceleration", "impact_score", "battery_status", "gps_status", "risk_label"]]
        return f"**Last 10 readings for {vehicle_id}:**\n\n{recent.to_markdown(index=False)}"

    if any(kw in q for kw in ["stats", "statistics", "summary", "overview"]):
        stats = compute_statistics(df, vehicle_id)
        resp = f"**Statistics for {vehicle_id}:**\n\n"
        for feat in ["speed", "acceleration", "impact_score", "battery_status"]:
            s = stats[feat]
            resp += f"**{feat}:** mean={s['mean']:.2f}, std={s['std']:.2f}, min={s['min']:.2f}, max={s['max']:.2f}\n\n"
        return resp

    if any(kw in q for kw in ["help", "what can", "commands"]):
        return ("**I can help with:**\n\n"
                "- 🚗 \"What is the current status?\"\n- ❓ \"Why was this alert triggered?\"\n"
                "- 🏎️ \"How fast is the vehicle?\"\n- 🔋 \"Battery level?\"\n"
                "- 📡 \"GPS status?\"\n- 💥 \"Any crashes?\"\n- 📊 \"Show stats\"\n- 📜 \"Show last readings\"")

    return (f"I'm the CATS Safety Bot monitoring **{vehicle_id}**. "
            "Try asking about status, alerts, speed, battery, GPS, impact, history, or stats. Type **help** for all commands.")


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

    # Status Banner
    render_status_banner(latest["risk_label"])
    if is_anomaly:
        st.markdown('<span class="anomaly-badge">🔮 ANOMALY DETECTED</span>', unsafe_allow_html=True)
    st.markdown("")

    # Gauge Row
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.plotly_chart(make_gauge(latest["speed"], "Speed", 0, 220,
                        SPEED_THRESHOLDS, "km/h"), use_container_width=True, key="g_spd")
    with g2:
        st.plotly_chart(make_gauge(latest["acceleration"], "Accel", 0, 15,
                        ACCELERATION_THRESHOLDS, "g"), use_container_width=True, key="g_acc")
    with g3:
        st.plotly_chart(make_gauge(latest["impact_score"], "Impact", 0, 1.0,
                        IMPACT_SCORE_THRESHOLDS), use_container_width=True, key="g_imp")
    with g4:
        st.plotly_chart(make_gauge(latest["battery_status"], "Battery", 0, 100,
                        BATTERY_THRESHOLDS, "%", reverse=True), use_container_width=True, key="g_bat")

    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("🏎️ Speed", f"{latest['speed']:.0f} km/h",
                   delta=f"{latest['speed'] - vehicle_df['speed'].mean():.0f} vs avg")
    with m2:
        st.metric("⚡ Acceleration", f"{latest['acceleration']:.1f} g")
    with m3:
        st.metric("💥 Impact", f"{latest['impact_score']:.3f}")
    with m4:
        st.metric("🔋 Battery", f"{latest['battery_status']:.0f}%")
    with m5:
        st.metric("📡 GPS", "🟢 Active" if latest["gps_status"] == 1 else "🔴 Lost")

    st.markdown("---")

    # Charts
    col_l, col_r = st.columns(2)
    with col_l:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Speed (km/h)", "Acceleration (g)"), vertical_spacing=0.12)
        fig.add_trace(go.Scatter(x=vehicle_df["timestamp"], y=vehicle_df["speed"],
                       mode="lines", name="Speed", line=dict(color="#3498db", width=2)), row=1, col=1)
        # Threshold lines
        fig.add_hline(y=SPEED_THRESHOLDS["warning"], line_dash="dash", line_color="#f39c12",
                      annotation_text="Warning", row=1, col=1)
        fig.add_hline(y=SPEED_THRESHOLDS["critical"], line_dash="dash", line_color="#e74c3c",
                      annotation_text="Critical", row=1, col=1)
        fig.add_trace(go.Scatter(x=vehicle_df["timestamp"], y=vehicle_df["acceleration"],
                       mode="lines", name="Acceleration", line=dict(color="#e74c3c", width=2)), row=2, col=1)
        fig.update_layout(height=400, showlegend=False, template="plotly_dark",
                          margin=dict(l=40, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Risk distribution
        risk_counts = vehicle_df["risk_label"].value_counts()
        colors = [STATUS_COLORS.get(label, "#999") for label in risk_counts.index]
        fig_pie = px.pie(values=risk_counts.values, names=risk_counts.index,
                         title="Risk Distribution", color_discrete_sequence=colors)
        fig_pie.update_layout(height=200, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

        # Battery + Impact
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=vehicle_df["timestamp"], y=vehicle_df["battery_status"],
                       mode="lines", name="Battery %", line=dict(color="#f39c12", width=2)))
        fig2.add_trace(go.Scatter(x=vehicle_df["timestamp"], y=vehicle_df["impact_score"] * 100,
                       mode="lines", name="Impact ×100", line=dict(color="#e74c3c", width=1, dash="dot")))
        fig2.update_layout(height=180, template="plotly_dark", margin=dict(l=40, r=20, t=10, b=20),
                           legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig2, use_container_width=True)


# ──────────────────────────────────────────────
#  PAGE: REAL-TIME SIMULATION
# ──────────────────────────────────────────────

def render_realtime(model, scaler, le, iso_forest, anom_scaler):
    st.markdown("### 🔴 Real-Time Telemetry Simulation")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        scenario = st.selectbox("Driving Scenario", ["mixed", "normal", "dangerous"])
    with col_ctrl2:
        interval = st.slider("Update Interval (sec)", 1, 5, 2)
    with col_ctrl3:
        st.markdown("")
        st.markdown("")
        running = st.toggle("▶️ Start Simulation", value=False)

    if "sim_history" not in st.session_state:
        st.session_state.sim_history = []
    if "sim_rng" not in st.session_state:
        st.session_state.sim_rng = np.random.default_rng(None)

    if running:
        reading = generate_sim_reading(st.session_state.sim_rng, scenario)
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
        # Keep last 50
        if len(st.session_state.sim_history) > 50:
            st.session_state.sim_history = st.session_state.sim_history[-50:]

        # Display current
        render_status_banner(pred_label)
        if is_anomaly:
            st.markdown('<span class="anomaly-badge">🔮 ANOMALY DETECTED (score: {:.3f})</span>'.format(anom_score),
                        unsafe_allow_html=True)
        st.markdown("")

        # Gauges
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            st.plotly_chart(make_gauge(reading["speed"], "Speed", 0, 220,
                            SPEED_THRESHOLDS, "km/h"), use_container_width=True, key="rt_spd")
        with g2:
            st.plotly_chart(make_gauge(reading["acceleration"], "Accel", 0, 15,
                            ACCELERATION_THRESHOLDS, "g"), use_container_width=True, key="rt_acc")
        with g3:
            st.plotly_chart(make_gauge(reading["impact_score"], "Impact", 0, 1.0,
                            IMPACT_SCORE_THRESHOLDS), use_container_width=True, key="rt_imp")
        with g4:
            st.plotly_chart(make_gauge(reading["battery_status"], "Battery", 0, 100,
                            BATTERY_THRESHOLDS, "%", reverse=True), use_container_width=True, key="rt_bat")

        # Live chart
        if len(st.session_state.sim_history) > 1:
            hist_df = pd.DataFrame(st.session_state.sim_history)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=("Speed (km/h)", "Impact Score"))
            colors = [STATUS_COLORS.get(r.get("risk_label", "NORMAL"), "#999") for r in st.session_state.sim_history]
            fig.add_trace(go.Scatter(x=hist_df["timestamp"], y=hist_df["speed"], mode="lines+markers",
                           marker=dict(color=colors, size=6), line=dict(color="#3498db")), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist_df["timestamp"], y=hist_df["impact_score"], mode="lines+markers",
                           marker=dict(color=colors, size=6), line=dict(color="#e74c3c")), row=2, col=1)
            fig.update_layout(height=350, template="plotly_dark", showlegend=False,
                              margin=dict(l=40, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

        # Explanation
        row = pd.Series(reading)
        explanations = generate_explanation(row)
        with st.expander("📋 Current Reading Explanation", expanded=True):
            for exp in explanations:
                st.markdown(f"- {exp}")
            st.markdown(f"**ML Confidence:** {pred_label} at {max(pred_probs.values()):.1%}")

        time.sleep(interval)
        st.rerun()

    else:
        if st.session_state.sim_history:
            st.info(f"Simulation paused. {len(st.session_state.sim_history)} readings captured.")
            hist_df = pd.DataFrame(st.session_state.sim_history)
            st.dataframe(hist_df, use_container_width=True)
        else:
            st.info("Toggle **Start Simulation** to begin real-time telemetry monitoring.")


# ──────────────────────────────────────────────
#  PAGE: ALERTS + TIMELINE
# ──────────────────────────────────────────────

def render_alerts(df, vehicle_id, model, scaler, le, iso_forest, anom_scaler):
    vehicle_df = df[df["vehicle_id"] == vehicle_id].sort_values("timestamp", ascending=False)
    alerts_df = vehicle_df[vehicle_df["risk_label"].isin([STATUS_WARNING, STATUS_CRITICAL])].head(30)

    if alerts_df.empty:
        st.success("✅ No active alerts for this vehicle.")
        return

    # ALERT TIMELINE
    st.markdown("### 📅 Alert Timeline")
    timeline_data = []
    for _, row in alerts_df.iterrows():
        timeline_data.append({
            "timestamp": row["timestamp"],
            "status": row["risk_label"],
            "speed": row["speed"],
        })
    tl_df = pd.DataFrame(timeline_data)
    color_map = {STATUS_WARNING: "#f39c12", STATUS_CRITICAL: "#e74c3c"}
    fig_tl = px.scatter(tl_df, x="timestamp", y="speed", color="status",
                        color_discrete_map=color_map, size_max=12,
                        title="Alert Timeline — Speed at Alert Time")
    fig_tl.update_layout(height=250, template="plotly_dark", margin=dict(l=40, r=20, t=40, b=20))
    fig_tl.update_traces(marker=dict(size=10))
    st.plotly_chart(fig_tl, use_container_width=True)

    st.markdown("---")

    # ALERT CARDS
    st.markdown(f"### 🚨 Alerts ({len(alerts_df)} most recent)")
    for idx, row in alerts_df.iterrows():
        status = row["risk_label"]
        emoji = STATUS_EMOJIS[status]
        explanations = generate_explanation(row)
        features = [row[col] for col in FEATURE_COLUMNS]
        pred_label, pred_probs = predict_risk(model, scaler, le, features)
        is_anomaly, anom_score = detect_anomaly(features, iso_forest, anom_scaler)

        anom_tag = " 🔮 ANOMALY" if is_anomaly else ""
        with st.expander(f"{emoji} {status}{anom_tag} — {row['timestamp']} | {row['speed']:.0f} km/h",
                         expanded=(status == STATUS_CRITICAL)):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("**Why this alert was generated:**")
                for exp in explanations:
                    st.markdown(f"- {exp}")
                if is_anomaly:
                    st.markdown(f"- 🔮 **Anomaly detected** — this reading deviates from normal driving patterns (score: {anom_score:.3f})")
            with c2:
                st.markdown("**ML Confidence:**")
                for label in [STATUS_CRITICAL, STATUS_WARNING, STATUS_NORMAL]:
                    if label in pred_probs:
                        st.progress(pred_probs[label], text=f"{label}: {pred_probs[label]:.1%}")

            ca, cb = st.columns([1, 3])
            with ca:
                if st.button("🚫 False Alarm", key=f"fa_{idx}"):
                    record_feedback(row, "false_alarm")
                    st.success("Marked as false alarm!")


# ──────────────────────────────────────────────
#  PAGE: AI CHAT
# ──────────────────────────────────────────────

def render_chat(df, vehicle_id, model, scaler, le):
    backend = get_backend()
    badge = {
        "groq": "🧠 Powered by Groq (Llama 3.3 70B)",
        "openrouter": "🧠 Powered by OpenRouter (Llama 3.3 70B)",
        "rule_based": "📋 Rule-based mode (set GROQ_API_KEY or OPENROUTER_API_KEY for AI chat)",
    }

    st.markdown("### 💬 Chat with CATS Safety Bot")
    st.markdown(f'<span class="ai-badge">{badge[backend]}</span>', unsafe_allow_html=True)
    st.caption(f"Ask anything about vehicle **{vehicle_id}**'s safety and telemetry.")

    if backend != "rule_based":
        with st.expander("⚙️ API Key Configuration"):
            st.markdown("""
            Set environment variables before launching:
            ```
            set GROQ_API_KEY=gsk_your_key_here
            set OPENROUTER_API_KEY=sk-or-your_key_here
            ```
            Get a free Groq key at [console.groq.com](https://console.groq.com)
            """)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": f"👋 Hi! I'm the CATS Safety Bot monitoring **{vehicle_id}**. Ask me anything about its telemetry, safety status, or alerts! {'🧠 AI-powered responses enabled.' if backend != 'rule_based' else ''}"}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about vehicle safety, telemetry, alerts..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Try LLM first, fallback to rules
        vehicle_df = df[df["vehicle_id"] == vehicle_id].sort_values("timestamp")
        latest = vehicle_df.iloc[-1] if len(vehicle_df) else pd.Series()
        vehicle_data = latest.to_dict() if len(latest) else {}
        stats = compute_statistics(df, vehicle_id)

        response = None
        if backend != "rule_based":
            with st.spinner("🧠 Thinking..."):
                response = llm_chat(st.session_state.messages, vehicle_data, stats)

        if response is None:
            response = handle_chat_rulebased(prompt, df, vehicle_id, model, scaler, le)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)


# ──────────────────────────────────────────────
#  PAGE: SIMULATOR + SHAP
# ──────────────────────────────────────────────

def render_simulator(model, scaler, le, iso_forest, anom_scaler):
    st.markdown("### 🎛️ Telemetry Simulator with AI Explainability")

    c1, c2 = st.columns(2)
    with c1:
        sim_speed = st.slider("Speed (km/h)", 0.0, 220.0, 60.0, 1.0)
        sim_accel = st.slider("Acceleration (g)", 0.0, 15.0, 1.0, 0.1)
        sim_impact = st.slider("Impact Score", 0.0, 1.0, 0.05, 0.01)
    with c2:
        sim_battery = st.slider("Battery (%)", 0.0, 100.0, 75.0, 1.0)
        sim_gps = st.selectbox("GPS Status", [1, 0], format_func=lambda x: "🟢 Active" if x == 1 else "🔴 Lost")

    features = [sim_speed, sim_accel, sim_impact, sim_battery, sim_gps]
    pred_label, pred_probs = predict_risk(model, scaler, le, features)
    is_anomaly, anom_score = detect_anomaly(features, iso_forest, anom_scaler)

    st.markdown("---")
    render_status_banner(pred_label)
    if is_anomaly:
        st.markdown(f'<span class="anomaly-badge">🔮 ANOMALY (score: {anom_score:.3f})</span>', unsafe_allow_html=True)
    st.markdown("")

    # Explanation
    sim_row = pd.Series({"speed": sim_speed, "acceleration": sim_accel,
                          "impact_score": sim_impact, "battery_status": sim_battery,
                          "gps_status": sim_gps})
    explanations = generate_explanation(sim_row)

    ex_col, shap_col = st.columns(2)

    with ex_col:
        st.markdown("**📋 Rule-Based Explanation:**")
        for exp in explanations:
            st.markdown(f"- {exp}")

        st.markdown("**📊 Prediction Confidence:**")
        for label in [STATUS_CRITICAL, STATUS_WARNING, STATUS_NORMAL]:
            if label in pred_probs:
                st.progress(pred_probs[label], text=f"{STATUS_EMOJIS.get(label, '')} {label}: {pred_probs[label]:.1%}")

    with shap_col:
        st.markdown("**🔬 SHAP Feature Contributions:**")
        try:
            import shap
            X_input = np.array(features).reshape(1, -1)
            X_scaled = scaler.transform(X_input)

            # Use KernelExplainer for broad model compatibility
            # Sample background data for KernelExplainer
            bg_data = shap.sample(pd.DataFrame(scaler.transform(load_data()[FEATURE_COLUMNS].values), columns=FEATURE_COLUMNS), 100)
            explainer = shap.KernelExplainer(model.predict_proba, bg_data)
            shap_values = explainer.shap_values(X_scaled, nsamples=100)

            # Get SHAP for predicted class
            pred_idx = list(le.classes_).index(pred_label)

            if isinstance(shap_values, list):
                sv = shap_values[pred_idx][0]
            else:
                sv = shap_values[0, :, pred_idx] if shap_values.ndim == 3 else shap_values[0]

            # Bar chart of SHAP values
            shap_df = pd.DataFrame({
                "Feature": FEATURE_COLUMNS,
                "SHAP Value": sv,
                "Abs SHAP": np.abs(sv),
            }).sort_values("Abs SHAP", ascending=True)

            colors = ["#e74c3c" if v > 0 else "#3498db" for v in shap_df["SHAP Value"]]
            fig_shap = go.Figure(go.Bar(
                x=shap_df["SHAP Value"], y=shap_df["Feature"],
                orientation="h", marker_color=colors,
                text=[f"{v:+.3f}" for v in shap_df["SHAP Value"]], textposition="outside",
            ))
            fig_shap.update_layout(
                height=250, template="plotly_dark",
                title=f"SHAP for → {pred_label}",
                xaxis_title="Impact on prediction",
                margin=dict(l=100, r=40, t=40, b=20),
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            st.caption("🔴 Red = pushes toward this class | 🔵 Blue = pushes away")
        except Exception as e:
            st.warning(f"SHAP visualization unavailable: {e}")


# ──────────────────────────────────────────────
#  PAGE: MODEL COMPARISON
# ──────────────────────────────────────────────

def render_model_comparison():
    st.markdown("### 🏆 Multi-Model Comparison")
    comparison = load_model_comparison()

    if comparison is None:
        st.warning("Run `python models/train_compare.py` to generate model comparison data.")
        return

    st.success(f"**Best Model: {comparison['best_model']}**")

    # Comparison table
    rows = []
    for name, metrics in comparison["models"].items():
        rows.append({
            "Model": ("🏆 " if name == comparison["best_model"] else "") + name,
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "F1 Score": f"{metrics['f1_weighted']:.4f}",
            "Precision": f"{metrics['precision_weighted']:.4f}",
            "Recall": f"{metrics['recall_weighted']:.4f}",
            "CV F1 (mean±std)": f"{metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}",
            "Train Time": f"{metrics['train_time_s']:.2f}s",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Bar chart comparison
    model_names = list(comparison["models"].keys())
    metrics_to_plot = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
    fig = go.Figure()
    colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]
    for i, metric in enumerate(metrics_to_plot):
        values = [comparison["models"][m][metric] for m in model_names]
        fig.add_trace(go.Bar(name=metric.replace("_weighted", "").title(),
                             x=model_names, y=values, marker_color=colors[i]))
    fig.update_layout(barmode="group", height=350, template="plotly_dark",
                      title="Model Performance Comparison", yaxis_range=[0.95, 1.0],
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
            fig_cm.update_layout(height=250, title=name, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_cm, use_container_width=True)


# ──────────────────────────────────────────────
#  PAGE: FEEDBACK + RETRAINING
# ──────────────────────────────────────────────

def render_feedback():
    st.markdown("### 📝 Feedback Log & Model Retraining")

    if os.path.exists(FEEDBACK_PATH):
        feedback_df = pd.read_csv(FEEDBACK_PATH)
        st.dataframe(feedback_df, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Feedback", len(feedback_df))
            st.metric("False Alarms", len(feedback_df[feedback_df["feedback_type"] == "false_alarm"]))
        with c2:
            csv = feedback_df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "feedback_log.csv", "text/csv")

        st.markdown("---")
        st.markdown("### 🔄 Retrain Model with Feedback")
        st.caption("Incorporate false alarm data to improve the classifier.")

        if st.button("🔄 Retrain Model", type="primary"):
            with st.spinner("Retraining model with feedback data..."):
                try:
                    # Load original data
                    df = pd.read_csv(DATASET_PATH)
                    false_alarms = feedback_df[feedback_df["feedback_type"] == "false_alarm"]

                    # For false alarms: downgrade the label
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
                        # Retrain
                        from sklearn.ensemble import RandomForestClassifier
                        from sklearn.preprocessing import StandardScaler, LabelEncoder
                        from sklearn.model_selection import train_test_split

                        X = df[FEATURE_COLUMNS].values
                        y = df["risk_label"].values
                        le_new = LabelEncoder()
                        y_enc = le_new.fit_transform(y)
                        scaler_new = StandardScaler()
                        X_scaled = scaler_new.fit_transform(X)

                        clf = RandomForestClassifier(n_estimators=100, max_depth=15,
                                                      class_weight="balanced", random_state=42, n_jobs=-1)
                        clf.fit(X_scaled, y_enc)

                        joblib.dump({"model": clf, "label_encoder": le_new}, CLASSIFIER_PATH)
                        joblib.dump(scaler_new, SCALER_PATH)

                        st.success(f"✅ Model retrained! {relabeled} labels adjusted from feedback.")
                        st.cache_resource.clear()
                    else:
                        st.info("No matching readings found to relabel. Model unchanged.")
                except Exception as e:
                    st.error(f"Retraining failed: {e}")
    else:
        st.info("No feedback recorded yet. Mark alerts as false alarms on the Alerts page to start building the feedback log.")


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────

def main():
    # Sidebar
    st.sidebar.markdown("## 🚗 CATS Safety Bot")
    st.sidebar.caption("Crisis Alert Tech System v2.0")
    st.sidebar.markdown("---")

    try:
        model, le, scaler = load_model()
        df = load_data()
        iso_forest, anom_scaler = load_anomaly_model()
    except Exception as e:
        st.error(f"Failed to load: {e}")
        st.info("Run: `python data/generate_dataset.py` → `python models/train.py`")
        return

    # Vehicle selector
    vehicle_ids = get_vehicle_ids(df)
    selected_vehicle = st.sidebar.selectbox("🚘 Vehicle", vehicle_ids)

    # LLM status
    backend = get_backend()
    backend_icon = {"groq": "🟢 Groq", "openrouter": "🟢 OpenRouter", "rule_based": "🟡 Rule-based"}
    st.sidebar.markdown(f"**AI Chat:** {backend_icon[backend]}")

    # Anomaly status
    st.sidebar.markdown(f"**Anomaly Detection:** {'🟢 Active' if iso_forest else '🔴 Off'}")

    st.sidebar.markdown("---")
    page = st.sidebar.radio("📍 Navigate", [
        "📊 Dashboard",
        "🔴 Real-Time Sim",
        "🚨 Alerts & Timeline",
        "💬 AI Chat",
        "🎛️ Simulator + SHAP",
        "🏆 Model Comparison",
        "📝 Feedback & Retrain",
    ])

    # Header
    st.title("🚗 CATS AI Safety Bot")
    st.caption("Crisis Alert Tech System — AI-Powered Vehicle Safety Monitoring")

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
    elif page == "📝 Feedback & Retrain":
        render_feedback()


if __name__ == "__main__":
    main()
