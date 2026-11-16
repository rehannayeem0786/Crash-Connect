# Architecture — CATS AI Safety Bot

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CATS AI Safety Bot                            │
│                   (Streamlit Web App)                            │
├─────────────┬───────────────┬──────────────┬───────────────────┤
│  Dashboard  │   Alerts      │    Chat      │    Simulator      │
│  (Plotly)   │   Panel       │   Interface  │   (Real-time      │
│             │               │              │    Prediction)    │
├─────────────┴───────────────┴──────────────┴───────────────────┤
│                    Core Engine                                   │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐     │
│  │ ML Classifier│  │ Explanation   │  │ Feedback         │     │
│  │ (Random      │  │ Engine        │  │ Recorder         │     │
│  │  Forest)     │  │ (Rules+SHAP)  │  │ (CSV Logger)     │     │
│  └──────┬───────┘  └───────────────┘  └──────────────────┘     │
│         │                                                        │
│  ┌──────┴──────────────────────────────────────────────────┐    │
│  │              Data Layer (Pandas)                         │    │
│  │  telemetry_data.csv  │  classifier.pkl  │  scaler.pkl   │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
1. Simulated Telemetry (CSV)
        │
        ▼
2. Feature Extraction (Pandas)
   [speed, acceleration, impact_score, battery_status, gps_status]
        │
        ▼
3. Standard Scaling (Scikit-learn)
        │
        ▼
4. Random Forest Classifier
        │
        ├──▶ Risk Label: NORMAL / WARNING / CRITICAL
        │
        ├──▶ Prediction Probabilities (confidence %)
        │
        └──▶ Feature Importances (explanation)
                │
                ▼
5. Rule-based Explanation Engine
   "Speed exceeded 140 km/h → CRITICAL"
        │
        ▼
6. Streamlit UI (Dashboard + Alerts + Chat)
        │
        └──▶ User Feedback (false alarm marking → CSV log)
```

## Component Details

### Data Generation (`data/generate_dataset.py`)
- Generates 8,000 simulated telemetry readings
- Realistic distributions for normal driving, warning, and critical scenarios
- 5 simulated vehicles (VH-1001 to VH-1005)
- Time-stamped over a 30-day period

### ML Pipeline (`models/train.py`)
- RandomForestClassifier with 100 trees
- StandardScaler for feature normalization
- 80/20 stratified train/test split
- 5-fold cross-validation
- Outputs: classifier.pkl, scaler.pkl, metrics.json

### Streamlit App (`app/app.py`)
- **Dashboard**: Real-time gauges, time-series charts, risk distribution
- **Alerts**: Warning/Critical alerts with expandable explanations
- **Chat**: Keyword-matching conversational interface
- **Simulator**: Interactive sliders for manual telemetry input
- **Feedback**: False alarm marking with CSV persistence

### Explanation Engine (`utils/telemetry.py`)
- Threshold-based rule explanations
- Maps feature values to human-readable safety descriptions
- Combined with ML confidence scores for transparent predictions
