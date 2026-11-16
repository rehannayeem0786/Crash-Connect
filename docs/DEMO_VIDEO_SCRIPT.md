# 🎬 3–5 Minute Demo Video Script — CATS AI Safety Bot

This script is structured to help you record a **flawless, professional, high-scoring 3–5 minute demo video** for the **CRASH CONNECT AI/ML Intern Practical Assessment**.

---

## ⏱️ Video Breakdown at a Glance

| Time | Scene | Focus Area |
|---|---|---|
| **0:00 – 0:40** | Introduction & Problem Overview | Objective, Architecture & Tech Stack |
| **0:40 – 1:30** | Live Dashboard & Telemetry Mapping | Gauges, Time-series, Pydeck GPS Track |
| **1:30 – 2:20** | Real-Time Telemetry Simulation | Dynamic streaming, live anomaly detection |
| **2:20 – 3:15** | Explainability & Simulator (SHAP) | Decision boundaries, SHAP marginal contributions |
| **3:15 – 3:55** | Alerts, Dossier Export & False Alarms | Forensic Incident Dossier, Feedback loop |
| **3:55 – 4:30** | Groq AI Conversational Assistant | Live telematics chat using Groq Qwen/Llama |
| **4:30 – 5:00** | Multi-Model Benchmark & Graphify | 99.94% Gradient Boosting vs RF vs SVM, Codebase Graph |

---

## 🎥 Scene-by-Scene Script & Voiceover

### Scene 1: Introduction (0:00 – 0:40)
- **On Screen:** Start on your browser showing the Streamlit App at `http://localhost:8501`, title bar: **🚗 CATS AI Safety Bot (Crisis Alert Tech System)**.
- **Voiceover:**
  > *"Hello CRASH CONNECT team! My name is [Your Name], and this is my submission for the AI/ML Intern Practical Assessment — the CATS AI Safety Bot prototype.*
  >
  > *CATS stands for Crisis Alert Tech System. Its mission is to be the 'Guardian on Wheels' — ingesting multi-sensor vehicle telemetry, detecting critical safety hazards in real-time, explaining why alerts occur using SHAP and domain rules, and offering a conversational AI interface powered by Groq LLMs.*
  >
  > *Let's dive into the live prototype."*

---

### Scene 2: Live Telemetry Dashboard & Geo-Tracking (0:40 – 1:30)
- **On Screen:** Navigate to **📊 Dashboard**. Select vehicle `VH-1001` or `VH-1003`.
- **Actions:**
  - Hover over the animated gauges (Speed, Acceleration, Impact, Battery).
  - Scroll down to the **🗺️ Live Route Telematics & Geolocation Tracking** Pydeck map.
  - Hover over green, orange, and red route points on the map.
- **Voiceover:**
  > *"Here on the main Dashboard, we monitor live vehicle telemetry. At the top, we have animated gauges tracking speed, g-force acceleration, impact severity, and battery level, alongside GPS link status.*
  >
  > *Below, we have integrated an interactive Pydeck telemetry map. Here you can see the vehicle's historical breadcrumb route through urban and highway corridors. Notice the color-coding: green points represent the normal driving envelope, orange indicates warning triggers like sudden deceleration, and red pins mark critical high-impact zones.*
  >
  > *Every reading is simultaneously evaluated by our supervised classifier and an unsupervised Isolation Forest anomaly detector."*

---

### Scene 3: Real-Time Telemetry Simulation (1:30 – 2:20)
- **On Screen:** Navigate to **🔴 Real-Time Sim**.
- **Actions:**
  - Switch scenario from `normal` to `dangerous` or `mixed`.
  - Toggle **▶️ Start Simulation**.
  - Show the gauges updating every 2 seconds and the live map plotting coordinates.
  - Show the `🔮 ANOMALY DETECTED` badge appearing on sudden spikes.
  - Pause the simulation.
- **Voiceover:**
  > *"In a production fleet environment, telemetry arrives as a continuous stream. In this Real-Time Simulation mode, the system dynamically simulates incoming CAN-bus sensor streams.*
  >
  > *As I toggle Start Simulation on the 'dangerous' scenario, notice how the speed gauge climbs past 140 km/h, g-force surges, and our model immediately transitions status to CRITICAL.*
  >
  > *Notice this purple badge: our Isolation Forest model flagged this event as an anomaly outlier with a negative decision score. Even if a vehicle encounters an unseen driving pattern, the dual-tier architecture flags it instantly."*

---

### Scene 4: Simulator & SHAP Explainability (2:20 – 3:15)
- **On Screen:** Navigate to **🎛️ Simulator + SHAP**.
- **Actions:**
  - Drag the **Speed** slider to 150 km/h, **Impact Score** to 0.75.
  - Show status turn red (CRITICAL).
  - Highlight the **🔬 SHAP Feature Attribution** bar chart on the right.
  - Point out which feature has the largest red bar.
- **Voiceover:**
  > *"A key requirement of the assessment is explainability: answering why a risk classification was made.*
  >
  > *Here in the Simulator, an engineer can adjust any parameter. On the left, our deterministic rule engine explains the physical violations — such as speed exceeding the 140 km/h critical threshold.*
  >
  > *On the right, we have implemented SHAP — Shapley Additive Explanations. This waterfall bar chart reveals the exact mathematical contribution of each sensor. The red bars demonstrate that the high impact score and speed drove the probability directly into the CRITICAL class. This transparent, glass-box approach is essential for automotive safety audits."*

---

### Scene 5: Alerts, Incident Dossier & False Alarms (3:15 – 3:55)
- **On Screen:** Navigate to **🚨 Alerts & Timeline**.
- **Actions:**
  - Scroll past the alert scatter timeline.
  - Expand a CRITICAL alert.
  - Click **📄 Generate Dossier** to display the terminal-style incident docket.
  - Click the **🚫 False Alarm** button to demonstrate feedback logging.
- **Voiceover:**
  > *"On the Alerts page, every warning and critical event is cataloged on an interactive timeline.*
  >
  > *For any incident, safety dispatchers can click 'Generate Dossier' with one touch. This produces an official Forensic Incident Investigation Dossier containing sensor telemetry, model consensus, GPS coordinates, and first responder action directives — downloadable as Markdown.*
  >
  > *Furthermore, if an alert was caused by a harmless bump, the operator can click 'False Alarm'. This feeds directly into our active learning feedback loop on the Feedback page, allowing the model to be retrained on corrected labels."*

---

### Scene 6: Groq AI Conversational Assistant (3:55 – 4:30)
- **On Screen:** Navigate to **💬 AI Chat**.
- **Actions:**
  - Point out the `🧠 Powered by Groq (qwen/qwen3.8-27b)` badge.
  - Click the quick-prompt button: **⚡ Is it safe to drive?**
  - Show the fast streaming response from Groq.
  - Type: *"What is the current vehicle status and why?"*
- **Voiceover:**
  > *"For the conversational interface, we integrated Groq's high-speed inference engine running Qwen 27B and Llama models.*
  >
  > *Notice that when I click 'Is it safe to drive?', the bot doesn't just give generic text — it inspects the vehicle's live telemetry dictionary injected into its context window, analyzes current speed, battery health, and GPS connectivity, and replies with actionable safety advice within milliseconds.*
  >
  > *If an API key is ever unavailable, the system seamlessly falls back to our offline rule-based NLP engine."*

---

### Scene 7: Multi-Model Benchmark & Graphify Codebase Graph (4:30 – 5:00)
- **On Screen:**
  - Quickly click **🏆 Model Comparison** to show the 3-model benchmark table (Gradient Boosting at 99.94% vs Random Forest vs SVM).
  - Click **🌐 Knowledge Graph (Graphify)** to show the interactive 2D/3D code network.
- **Voiceover:**
  > *"To validate our ML choice, we benchmarked three architectures across 5-fold stratified cross-validation. Gradient Boosting achieved 99.94% test accuracy and a 0.9978 cross-validated F1-score, outperforming Random Forest and SVM.*
  >
  > *Finally, we used Graphify to extract the repository's AST call graph and community clusters into an interactive network visualization directly inside the app.*
  >
  > *All 12 unit tests pass, documentation is comprehensive, and the code is structured cleanly. Thank you for your time, and I look forward to contributing to CRASH CONNECT!"*

---

## 💡 Pro-Tips for Recording

1. **Resolution:** Record in 1080p (1920x1080) at 60 or 30 FPS using OBS Studio, Loom, or Windows Xbox Game Bar (`Win + Alt + R`).
2. **Audio:** Use a headset or external mic so your voice is crisp and clear without background noise.
3. **Cursor:** Keep mouse movements smooth and intentional.
4. **Pacing:** Don't rush; speak with confidence as if presenting to senior engineering leadership.
5. **Pre-check:** Ensure your `.env` has the `GROQ_API_KEY` active so the AI chat responds instantly.
