# Graph Report - CRASH CONNECT  (2026-11-16)

## Corpus Check
- Corpus is ~9,085 words - fits in a single context window. You may not need a graph.

## Summary
- 141 nodes · 310 edges · 14 communities (6 shown, 8 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Streamlit UI & Dashboard
- Telemetry Simulation & Data Generation
- Groq LLM & Conversational Engine
- Explanation Validation & Tests
- Model Serialization & Caching
- Model Training Pipeline
- Model Inference Unit Tests
- Dataset Schema Unit Tests

## God Nodes (most connected - your core abstractions)
1. `main()` - 14 edges
2. `generate_explanation()` - 12 edges
3. `render_realtime()` - 11 edges
4. `render_simulator()` - 11 edges
5. `render_dashboard()` - 9 edges
6. `render_alerts()` - 9 edges
7. `generate_dataset()` - 9 edges
8. `render_model_comparison()` - 8 edges
9. `predict_risk()` - 7 edges
10. `render_chat()` - 7 edges

## Surprising Connections (you probably didn't know these)
- `handle_chat_rulebased()` --calls--> `generate_explanation()`  [EXTRACTED]
  app/app.py → utils/telemetry.py
- `render_realtime()` --calls--> `generate_explanation()`  [EXTRACTED]
  app/app.py → utils/telemetry.py
- `render_alerts()` --calls--> `generate_explanation()`  [EXTRACTED]
  app/app.py → utils/telemetry.py
- `render_simulator()` --calls--> `generate_explanation()`  [EXTRACTED]
  app/app.py → utils/telemetry.py
- `main()` --calls--> `get_backend()`  [EXTRACTED]
  app/app.py → utils/llm.py

## Import Cycles
- None detected.

## Communities (14 total, 8 thin omitted)

### Community 0 - "Streamlit UI & Dashboard"
Cohesion: 0.24
Nodes (15): detect_anomaly(), load_data(), load_model_comparison(), main(), make_gauge(), predict_risk(), record_feedback(), render_alerts() (+7 more)

### Community 2 - "Telemetry Simulation & Data Generation"
Cohesion: 0.16
Nodes (7): generate_sim_reading(), classify_reading(), generate_critical_scenarios(), generate_dataset(), generate_normal_driving(), generate_warning_scenarios(), main()

### Community 3 - "Groq LLM & Conversational Engine"
Cohesion: 0.16
Nodes (8): handle_chat_rulebased(), render_chat(), _build_context(), chat(), chat_with_groq(), chat_with_openrouter(), get_backend(), compute_statistics()

### Community 4 - "Explanation Validation & Tests"
Cohesion: 0.15
Nodes (5): TestExplanations, generate_explanation(), get_latest_reading(), get_vehicle_history(), get_vehicle_ids()

### Community 5 - "Model Serialization & Caching"
Cohesion: 0.15
Nodes (6): load_anomaly_model(), load_model(), detect_anomaly(), train_anomaly_detector(), evaluate(), train_and_compare()

### Community 6 - "Model Training Pipeline"
Cohesion: 0.29
Nodes (4): load_data(), main(), save_model(), train_model()

## Knowledge Gaps
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TestModel` connect `Model Inference Unit Tests` to `Evaluation & Anomaly Detection`?**
  _High betweenness centrality (0.094) - this node is a cross-community bridge._
- **Why does `TestDataset` connect `Dataset Schema Unit Tests` to `Evaluation & Anomaly Detection`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **Why does `generate_explanation()` connect `Explanation Validation & Tests` to `Streamlit UI & Dashboard`, `Evaluation & Anomaly Detection`, `Groq LLM & Conversational Engine`?**
  _High betweenness centrality (0.075) - this node is a cross-community bridge._