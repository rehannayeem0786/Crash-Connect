# Data Directory

## telemetry_data.csv

Simulated vehicle telemetry dataset with 8,000 readings across 5 monitored assets (`VH-1001` through `VH-1005`).

### Schema & Feature Dictionary

| Column | Type | Description | Operating Range |
|---|---|---|---|
| `timestamp` | datetime | Timestamp of telemetry packet transmission | Oct 2026 |
| `vehicle_id` | string | Monitored asset identifier | VH-1001 to VH-1005 |
| `speed` | float | Vehicle velocity in km/h | 0.0 – 220.0 km/h |
| `acceleration` | float | G-force accelerometer load | 0.0 – 15.0 g |
| `impact_score` | float | Sensor impact severity index (0 = none, 1.0 = collision) | 0.0 – 1.0 |
| `battery_status` | float | Battery pack state-of-charge percentage | 0.0 – 100.0% |
| `gps_status` | int | GPS satellite link connection flag (1 = Active, 0 = Signal Lost) | 0 or 1 |
| `latitude` | float | Vehicle geospatial latitude coordinate | ~12.80 – 13.15 °N |
| `longitude` | float | Vehicle geospatial longitude coordinate | ~77.50 – 77.85 °E |
| `risk_label` | string | Ground truth risk classification | NORMAL, WARNING, CRITICAL |

### Risk Class Distribution

- **NORMAL** (~61.4%): Baseline driving envelope; all parameters within nominal operating bounds.
- **WARNING** (~14.1%): Single parameter exceeds advisory threshold (e.g., speed > 100 km/h, braking > 4 g, battery < 30%, or GPS loss).
- **CRITICAL** (~24.6%): Collision-level impact (> 0.7), rollover/extreme g-force (> 7 g), excessive velocity (> 140 km/h), or critical battery depletion (< 10%).
