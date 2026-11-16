# Data Directory

## telemetry_data.csv

Simulated vehicle telemetry dataset with 8,000 readings.

### Columns

| Column | Type | Description | Range |
|---|---|---|---|
| `timestamp` | datetime | When the reading was recorded | Oct 2026 |
| `vehicle_id` | string | Unique vehicle identifier | VH-1001 to VH-1005 |
| `speed` | float | Vehicle speed in km/h | 0 – 220 |
| `acceleration` | float | Acceleration in g-force | 0 – 15 |
| `impact_score` | float | Impact severity (0 = none, 1 = max) | 0.0 – 1.0 |
| `battery_status` | float | Battery level percentage | 0 – 100 |
| `gps_status` | int | GPS signal (1 = active, 0 = lost) | 0 or 1 |
| `risk_label` | string | Classified risk level | NORMAL, WARNING, CRITICAL |

### Distribution

- **NORMAL** (~60%): Regular driving, all parameters within safe limits
- **WARNING** (~25%): One parameter exceeds a warning threshold
- **CRITICAL** (~15%): One or more parameters exceed critical thresholds, or multiple warning-level flags
