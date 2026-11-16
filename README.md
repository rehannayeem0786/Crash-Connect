# 🚗 CATS AI Safety Bot v2.1 (Top 1 Edition)

**Crisis Alert Tech System — AI-Powered Vehicle Telemetry Analysis & Safety Monitoring**

A production-grade prototype AI safety assistant that analyses vehicle telemetry, identifies safety risks, maps GPS telematics routes, explains alerts with SHAP attribution, exports forensic incident dossiers, and answers questions using Groq conversational AI.

Built as part of the CRASH CONNECT AI/ML Internship Assessment.

---

## 🎯 Submission Kit

| Resource | Document | Purpose |
|---|---|---|
| 🎬 **Demo Video Script** | [docs/DEMO_VIDEO_SCRIPT.md](docs/DEMO_VIDEO_SCRIPT.md) | Timed 3–5 min scene-by-scene recording walkthrough |
| 📝 **Assessment Form Answers** | [docs/SUBMISSION_ANSWERS.md](docs/SUBMISSION_ANSWERS.md) | Comprehensive responses ready for Google Form submission |
| 🏗️ **Architecture Document** | [docs/architecture.md](docs/architecture.md) | Full data flow & system design |
| 🔬 **ML Methodology** | [docs/ml_approach.md](docs/ml_approach.md) | Multi-model benchmarking, SHAP, and Anomaly Detection |

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 **Dashboard & Gauges** | Real-time animated gauges, delta indicators, and risk distribution |
| 🗺️ **GPS Route Telematics** | Interactive Pydeck map showing vehicle breadcrumb trails & incident spots |
| 🔴 **Real-Time Simulation** | Dynamic streaming telemetry generation with instant ML inference |
| 🚨 **Alerts & Timeline** | Severity-colored scatter timeline with expandable alert cards |
| 📄 **Forensic Incident Dossier** | 1-click official incident docket export with action protocols |
| 💬 **AI Conversational Chat** | Groq API (Qwen 27B / Llama 3.3) with quick-prompt chips |
| 🎛️ **Simulator + SHAP** | Interactive sliders with mathematical SHAP attribution plots |
| 🏆 **Multi-Model Benchmark** | Side-by-side Gradient Boosting (99.9%) vs RF vs SVM |
| 🔮 **Anomaly Detection** | Unsupervised Isolation Forest catching 95.2% of critical anomalies |
| 🌐 **Codebase Graph (Graphify)** | Interactive 2D/3D network topology and architectural community clusters |
| 📝 **Feedback & Retraining** | False alarm audit trail and active learning model retraining |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     CATS AI Safety Bot v2.0                          │
│                      (Streamlit Web App)                             │
├──────────┬───────────┬──────────┬───────────┬──────────┬────────────┤
│Dashboard │ Real-Time │  Alerts  │  AI Chat  │Simulator │  Model     │
│ (Gauges) │ Simulation│ Timeline │(Groq/OR)  │ + SHAP   │ Comparison │
├──────────┴───────────┴──────────┴───────────┴──────────┴────────────┤
│                        Core Engine                                   │
│ ┌───────────────┐ ┌──────────────┐ ┌────────────┐ ┌──────────────┐  │
│ │ Gradient      │ │ Isolation    │ │ SHAP       │ │ Groq /       │  │
│ │ Boosting      │ │ Forest       │ │ Explainer  │ │ OpenRouter   │  │
│ │ Classifier    │ │ Anomaly Det. │ │            │ │ LLM Chat     │  │
│ └───────────────┘ └──────────────┘ └────────────┘ └──────────────┘  │
│ ┌────────────────────────────────────────────────────────────────┐   │
│ │                  Data Layer (Pandas)                           │   │
│ │ telemetry_data.csv │ classifier.pkl │ anomaly_detector.pkl     │   │
│ └────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+
- (Optional) Groq API key for AI chat — [Get free key](https://console.groq.com)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/cats-ai-safety-bot.git
cd cats-ai-safety-bot

# 2. Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate dataset
python data/generate_dataset.py

# 5. Train ML models (single model)
python models/train.py

# 6. Train multi-model comparison
python models/train_compare.py

# 7. Train anomaly detector
python models/anomaly.py

# 8. (Optional) Set API keys for AI chat
set GROQ_API_KEY=gsk_your_key_here
# Or:
set OPENROUTER_API_KEY=sk-or-your_key_here

# 9. Launch the app
streamlit run app/app.py
```

The app opens at **http://localhost:8501**.

---

## 📂 Project Structure

```
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── .gitignore
│
├── app/
│   └── app.py                        # Main Streamlit app (7 pages)
│
├── data/
│   ├── generate_dataset.py           # Telemetry data generator
│   ├── telemetry_data.csv            # 8,000-row dataset
│   └── README.md                     # Data dictionary
│
├── models/
│   ├── train.py                      # Single model training
│   ├── train_compare.py              # Multi-model comparison
│   ├── evaluate.py                   # Evaluation plots
│   ├── anomaly.py                    # Isolation Forest training
│   ├── classifier.pkl                # Best model (Gradient Boosting)
│   ├── scaler.pkl                    # Feature scaler
│   ├── anomaly_detector.pkl          # Isolation Forest model
│   ├── model_comparison.json         # Comparison metrics
│   └── metrics.json                  # Evaluation metrics
│
├── utils/
│   ├── config.py                     # Thresholds & configuration
│   ├── telemetry.py                  # Data processing & explanations
│   └── llm.py                        # Groq/OpenRouter chat integration
│
├── tests/
│   └── test_classifier.py            # Unit tests (12 tests)
│
└── docs/
    ├── architecture.md               # System architecture
    └── ml_approach.md                 # ML methodology & evaluation
```

---

## 🤖 ML Approach

### Multi-Model Comparison

| Model | Accuracy | F1 | CV F1 |
|---|---|---|---|
| **🏆 Gradient Boosting** | **99.94%** | **99.94%** | **99.78%** |
| Random Forest | 99.31% | 99.31% | 99.57% |
| SVM (RBF) | 97.12% | 97.18% | 97.37% |

### Explainability Stack
1. **SHAP waterfall plots** — per-prediction feature contributions
2. **Rule-based explanations** — human-readable threshold checks
3. **Feature importance** — global model-level ranking

### Anomaly Detection
- Isolation Forest trained on NORMAL data only
- Catches 95.2% of CRITICAL readings as anomalous

See [docs/ml_approach.md](docs/ml_approach.md) for full details.

---

## 💬 AI Chat

The chatbot supports:
- **AI Mode** (Groq/OpenRouter) — natural language understanding with live telemetry context
- **Rule Mode** — keyword-matching fallback (no API key needed)

Example questions:
- "What is the current vehicle status?"
- "Why was this alert triggered?"
- "Is it safe to drive?"
- "Show me the speed history"
- "What does a high impact score mean?"

---

## ⚠️ Limitations

1. Simulated data only — not trained on real vehicle telemetry
2. No temporal modeling — each reading classified independently
3. Simplified features — real vehicles have hundreds of CAN-bus signals
4. AI chat requires API key — falls back to rule-based without one

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Core | Python 3.11, Pandas, NumPy |
| ML | Scikit-learn (RF, GB, SVM, Isolation Forest) |
| Explainability | SHAP |
| Chat AI | Groq API / OpenRouter (Llama 3.3 70B) |
| Web UI | Streamlit |
| Visualization | Plotly |
| Serialization | Joblib |
