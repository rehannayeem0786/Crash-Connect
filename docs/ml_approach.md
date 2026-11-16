# ML Approach — CATS AI Safety Bot

## Model Selection: Multi-Model Comparison

We trained and compared three classifiers to find the best model:

### Results

| Model | Accuracy | F1 Score | CV F1 (5-fold) | Train Time |
|---|---|---|---|---|
| **🏆 Gradient Boosting** | **99.94%** | **99.94%** | **99.78% ± 0.12%** | 4.18s |
| Random Forest | 99.31% | 99.31% | 99.57% ± 0.16% | 0.38s |
| SVM (RBF) | 97.12% | 97.18% | 97.37% ± 0.33% | 0.58s |

**Winner: Gradient Boosting** — highest accuracy and cross-validated F1 score.

### Why Gradient Boosting?

1. **Best accuracy**: 99.94% on test set (only 1 misclassification in 1,600 samples)
2. **Sequential learning**: Each tree corrects errors from the previous one
3. **Feature importance**: Directly available for explainability
4. **Robust**: Handles class imbalance well with appropriate hyperparameters

### Why Not The Others?

- **Random Forest**: Strong runner-up (99.31%), faster to train, but slightly less accurate
- **SVM (RBF)**: Good general classifier (97.12%), but weaker on the WARNING class (86% precision)

## Anomaly Detection Layer

In addition to the classifier, we use an **Isolation Forest** for unsupervised anomaly detection:

- Trained only on NORMAL driving data (4,908 readings)
- Flags readings that deviate from normal patterns
- Results on full dataset:
  - NORMAL → 5.0% flagged as anomalous (expected false positive rate)
  - WARNING → 45.4% flagged as anomalous
  - **CRITICAL → 95.2% flagged as anomalous** ✅

This dual-layer approach (supervised classifier + unsupervised anomaly detector) provides defense in depth.

## Explainability: Three Layers

1. **SHAP Values** — Per-prediction feature contribution plots. Shows exactly which features pushed the model toward each class. Interactive waterfall charts in the Simulator page.

2. **Rule-Based Explanations** — Human-readable threshold checks ("Speed exceeded 140 km/h → CRITICAL"). Always available, even offline.

3. **Feature Importance** — Global model-level importance ranking across all predictions.

## AI-Powered Conversational Interface

The chatbot supports two modes:

| Mode | Backend | Description |
|---|---|---|
| **AI Mode** | Groq (Llama 3.3 70B) or OpenRouter | Full natural language understanding with live telemetry context |
| **Rule Mode** | Keyword matching | Offline fallback with pattern-matched responses |

In AI mode, the LLM receives the vehicle's current telemetry data as context and can answer open-ended questions like:
- "Is this vehicle safe to drive?"
- "What should the driver do right now?"
- "Compare the current reading to the vehicle's history"

## Real-Time Simulation

The app includes a real-time simulation mode that:
- Generates new telemetry readings every 1-5 seconds
- Classifies each reading instantly with the ML model
- Runs anomaly detection on each reading
- Supports three driving scenarios: normal, dangerous, mixed
- Displays live-updating gauges and charts

## Training Process

### Data
- **8,000 simulated telemetry readings** across 5 vehicles
- **Distribution**: NORMAL (61.4%), CRITICAL (24.6%), WARNING (14.1%)
- **Features**: speed, acceleration, impact_score, battery_status, gps_status

### Pipeline
1. Standard scaling (zero mean, unit variance)
2. 80/20 stratified train/test split
3. 5-fold stratified cross-validation
4. Train three models, auto-select the best
5. Train Isolation Forest on NORMAL data for anomaly detection
6. Generate SHAP values for per-prediction explainability

### Feedback Loop
Users can mark alerts as false alarms. The feedback is logged and can be used to retrain the model, adjusting labels based on user corrections. This demonstrates an iterative ML pipeline.

## Limitations

1. **Simulated data** — not trained on real vehicle telemetry
2. **No temporal modeling** — each reading classified independently
3. **Simplified features** — real vehicles have hundreds of CAN-bus signals
4. **No real-time streaming** — simulation generates synthetic data, not live feeds
5. **AI chat requires API key** — falls back to rule-based without Groq/OpenRouter key

## AI Tools Used

- **Scikit-learn** — Model training (Random Forest, Gradient Boosting, SVM, Isolation Forest)
- **SHAP** — Per-prediction explainability
- **Groq API / OpenRouter** — LLM-powered conversational interface
- **Streamlit** — Web interface with real-time simulation
- **Plotly** — Interactive charts, gauges, and timelines
- **Pandas/NumPy** — Data processing and generation
