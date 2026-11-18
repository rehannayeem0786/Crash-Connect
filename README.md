# 🚗 CATS AI Safety Bot — Crisis Alert Tech System

**Autonomous Vehicle Telematics Analysis & Real-Time Safety Risk Monitoring**

A production-grade prototype AI safety assistant that ingests multi-sensor vehicle telemetry, classifies safety risk in real time (`NORMAL`, `WARNING`, `CRITICAL`), isolates abnormal sensor spikes via unsupervised anomaly detection, provides mathematical SHAP explainability, maps geospatial routes, and answers operator inquiries through conversational AI.

Built for the **CRASH CONNECT AI/ML Internship Practical Assessment** — *Guardian on Wheels*.

---

## 🌟 Key Capabilities

| Capability | Technical Implementation | Impact |
|---|---|---|
| **Multi-Model Risk Classifier** | Gradient Boosting (Production) vs Random Forest vs SVM | **99.94% Accuracy** and **0.9978 CV F1** on holdout test set |
| **Unsupervised Anomaly Layer** | Isolation Forest trained strictly on nominal driving envelope | Detects **95.2% of Critical incidents** as multivariate outliers |
| **Dual-Tier Explainability** | SHAP (`KernelExplainer`) attribution + Deterministic physical rule engine | Transparent directional log-odds marginal contributions per sensor |
| **Conversational Safety AI** | Groq API (`qwen/qwen3.8-27b`) with live telemetry context injection | Sub-second conversational vehicle safety analysis with offline fallback |
| **Geospatial Route Mapping** | Interactive Pydeck 3D telematics map | Visualizes vehicle trails, speed zones, incident pins, and GPS dead zones |
| **Incident Investigation Dossier** | One-click forensic docket generator | Formatted emergency dispatch briefs with sensor snapshots and action protocols |
| **Active Learning Feedback Loop** | Operator false alarm tagging and dynamic model retraining | Continuous active learning pipeline that adjusts model decision weights |
| **Codebase Knowledge Graph** | Graphify AST extraction & community clustering | Embedded 2D/3D interactive network mapping all project code and call hierarchies |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                  CATS AI SAFETY BOT — MISSION CONTROL (STREAMLIT)            │
├─────────────┬──────────────┬──────────────┬─────────────┬────────────────────┤
│  Dashboard  │  Real-Time   │   Alerts &   │   AI Chat   │ Simulator + SHAP   │
│  & HUD      │  Simulation  │   Timeline   │  (Groq AI)  │ Attribution        │
├─────────────┴──────────────┴──────────────┴─────────────┴────────────────────┤
│                        DUAL-TIER AI INFERENCE ENGINE                         │
│  ┌─────────────────────────┐  ┌───────────────────────┐  ┌─────────────────┐ │
│  │ Gradient Boosting       │  │ Isolation Forest      │  │ SHAP Game-      │ │
│  │ Classifier (Supervised) │  │ Anomaly Core          │  │ Theoretic Factor│ │
│  │ 99.94% Test Accuracy    │  │ (Unsupervised Outlier)│  │ Attribution     │ │
│  └────────────┬────────────┘  └───────────┬───────────┘  └────────┬────────┘ │
│               └─────────────┬─────────────┘                       │          │
│                             ▼                                     ▼          │
│                [Consensus Verdict: NORMAL / WARNING / CRITICAL]              │
│                             │                                                │
│                             ▼                                                │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  Deterministic Rule Explanations & Forensic Dossier Generator          │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────────────────────────┤
│                         TELEMETRY DATA & ASSET LAYER                         │
│  telemetry_data.csv (8,000 Readings) │ classifier.pkl │ anomaly_detector.pkl │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation & Quick Start

### Prerequisites
* Python 3.10 or 3.11
* (Optional) Free Groq API Key from [console.groq.com](https://console.groq.com) for AI conversational features.

### Step 1: Clone the Repository
```bash
git clone https://github.com/<your-username>/crash-connect-cats.git
cd crash-connect-cats
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows:
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables (Optional for AI Chat)
Copy the example environment file and add your Groq API key:
```bash
# Windows:
copy .env.example .env

# macOS / Linux:
cp .env.example .env
```
Open `.env` and paste your key:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```
*(Note: If no API key is provided, the chat automatically operates in offline rule-based mode).*

### Step 5: Launch the Application
```bash
streamlit run app/app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

> **Note:** Pre-trained production model weights (`models/classifier.pkl`, `models/anomaly_detector.pkl`) and the full 8,000-reading dataset are bundled in the repository. You can launch and demo immediately without waiting for training.

---

## 📂 Project Structure

```
├── README.md                         # Master technical documentation
├── requirements.txt                  # Python dependencies
├── .env.example                      # Template for API keys
├── .gitignore                        # Git ignore specifications
│
├── .streamlit/
│   └── config.toml                   # Executive dark cockpit styling theme
│
├── app/
│   ├── __init__.py
│   └── app.py                        # 8-page Streamlit telematics cockpit
│
├── data/
│   ├── __init__.py
│   ├── README.md                     # Schema and feature dictionary
│   ├── generate_dataset.py           # Parametric telemetry simulation script
│   └── telemetry_data.csv            # 8,000 synthetic telematics observations
│
├── models/
│   ├── __init__.py
│   ├── train.py                      # Single-model baseline training script
│   ├── train_compare.py              # Multi-model benchmarking script (GB, RF, SVM)
│   ├── evaluate.py                   # Plot generator (confusion matrix, feature imp.)
│   ├── anomaly.py                    # Isolation Forest anomaly detector training
│   ├── classifier.pkl                # Production Gradient Boosting model
│   ├── scaler.pkl                    # Feature standard scaler
│   ├── anomaly_detector.pkl          # Trained Isolation Forest model
│   ├── anomaly_scaler.pkl            # Anomaly feature scaler
│   ├── model_comparison.json         # Cross-validation benchmark metrics
│   ├── metrics.json                  # Baseline evaluation metrics
│   ├── confusion_matrix.png          # Confusion matrix visualization
│   └── feature_importance.png        # Global feature importance plot
│
├── utils/
│   ├── __init__.py
│   ├── config.py                     # Physical thresholds & safety bounds
│   ├── telemetry.py                  # Statistical processors & explanation engine
│   └── llm.py                        # Groq LLM integration with prompt grounding
│
├── tests/
│   ├── __init__.py
│   └── test_classifier.py            # 12 automated unit tests (Pytest)
│
└── graphify-out/
    ├── GRAPH_REPORT.md               # Codebase audit & community detection report
    ├── graph.html                    # 2D/3D interactive network knowledge graph
    └── graph.json                    # AST dependency extraction graph data
```

---

## 🤖 Machine Learning Methodology & Benchmarks

### 1. Multi-Model Architecture Comparison
To avoid heuristic bias, three distinct classification paradigms were trained and evaluated on an 80/20 stratified split reinforced by **5-Fold Stratified Cross-Validation**:

| Architecture | Test Accuracy | Weighted F1 | 5-Fold CV F1 Score | Training Latency |
|---|---|---|---|---|
| **🏆 Gradient Boosting (Production)** | **99.94%** | **0.9994** | **0.9978 ± 0.0012** | 4.18s |
| Random Forest (Bagging Baseline) | 99.31% | 0.9931 | 0.9957 ± 0.0016 | 0.38s |
| Support Vector Machine (RBF Kernel) | 97.12% | 0.9718 | 0.9737 ± 0.0033 | 0.58s |

* **Why Gradient Boosting Won:** Gradient Boosting sequentially optimizes decision trees over pseudo-residuals, creating sharper boundaries around complex multi-sensor warning states (such as simultaneous moderate speed combined with hard braking).
* It achieved near-flawless generalization: **only 1 misclassification out of 1,600 holdout samples**.

### 2. Dual-Tier Outlier Defense (Isolation Forest)
Supervised models cannot reliably flag unseen edge cases. To protect vehicle occupants against zero-day failures, an unsupervised **Isolation Forest** was trained exclusively on nominal driving data (4,908 `NORMAL` samples with 5% contamination).
* **Validation Outcome:** When tested across the full dataset, the Isolation Forest successfully isolated **95.2% of Critical emergency readings** as multivariate anomalies, providing defense-in-depth.

### 3. Explainability Stack
* **SHAP (KernelExplainer):** Computes game-theoretic marginal contributions per feature for each prediction. Directional waterfall bars highlight whether a sensor pushed the probability towards or away from danger.
* **Deterministic Threshold Engine:** Generates plain-English explanations mapped against physical vehicle safety limits (e.g., *Speed > 140 km/h, G-force > 7.0 g, Impact > 0.7*).

---

## 🗺️ Application Tour (8 Interactive Pages)

1. **📊 Dashboard**: Heads-Up Display (HUD) top bar, glassmorphic metric cards, animated Plotly gauges, Pydeck 3D GPS route map, and risk envelope pie charts.
2. **🔴 Real-Time Sim**: Dynamic vehicle telemetry streaming generator with moving map updates and instant anomaly alerts across Normal, Dangerous, and Mixed scenarios.
3. **🚨 Alerts & Timeline**: Scatter incident timeline, expandable alert cards, false alarm tagging, and one-click **Forensic Incident Dossier** generator.
4. **💬 AI Chat**: Conversational safety assistant powered by Groq Qwen 27B / Llama models with quick-prompt chips.
5. **🎛️ Simulator + SHAP**: Interactive slider sandbox with real-time SHAP factor attribution plots.
6. **🏆 Model Comparison**: Detailed performance matrix comparing Gradient Boosting, Random Forest, and SVM with confusion matrices.
7. **🌐 Knowledge Graph (Graphify)**: Embedded interactive 2D/3D codebase network mapping all 24 project files and AST call hierarchies.
8. **📝 Feedback & Retrain**: Operator false alarm audit log and active learning pipeline that adjusts model weights on user feedback.

---

## 🧪 Automated Testing

The project includes an automated test suite of **12 unit tests** in `tests/test_classifier.py`:
```bash
pytest tests/test_classifier.py -v
```
**Test Coverage:**
* ✅ Dataset schema, feature ranges, non-negative bounds, and class distributions.
* ✅ Model serialization, label inverse-transformation, and inference stability.
* ✅ Positive identification of nominal driving envelopes.
* ✅ Positive identification of extreme critical collision scenarios.
* ✅ Deterministic explanation rule generation.

---

## ⚠️ Limitations & Production Scaling Roadmap

### Current Prototype Limitations
1. **Simulated Telematics:** Synthetic data abstracts real-world sensor noise, voltage fluctuations, and weather degradation.
2. **Discrete Time Slices:** Observations are evaluated independently without recurrent temporal sequence modeling (e.g., LSTMs or Temporal Transformers).
3. **Feature Dimensionality:** Modern vehicles expose 500+ PID parameters via OBD-II/CAN-bus, abstracted here into 5 core indicators.

### Roadmap to Fleet Production
1. **Edge Inference:** Compile the Gradient Boosting model to ONNX runtime for sub-10ms execution on vehicle Telematics Control Units (TCUs / ARM Cortex).
2. **Streaming Pipeline:** Replace batch CSV polling with Apache Kafka / MQTT broker ingesting telemetry at 50 Hz.
3. **Temporal Trajectory Windows:** Implement a rolling 10-second Temporal Convolutional Network (TCN) to classify progressive mechanical wear before acute failures occur.

---

## 📜 License & Compliance

Developed for the **CRASH CONNECT AI/ML Internship Practical Assessment**.
All simulated data and algorithms are intended for prototype evaluation.
