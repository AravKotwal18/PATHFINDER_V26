# PATHFINDER V26

**IMU-only trajectory reconstruction using the Sentinel-6D adaptive inertial estimator, with an output validator and a three-view playback dashboard.**

| | |
|---|---|
| **Team** | Agastaya Jindal (Leader), Arav Kotwal, Aditya Ranjan, Yogesh Jain |
| **Language** | Python |
| **Platform** | Windows (PowerShell) |
| **Status** | Final submission package |

This repository contains the final presentation and only the files needed to run the estimator, validate its output, and play the included 15-second demo.

---

## Contents

- [Repository structure](#repository-structure)
- [Quick start](#quick-start)
- [Output format](#output-format)
- [Dataset](#dataset)
- [Design notes](#design-notes)
- [Scope and limitations](#scope-and-limitations)

---

## Repository structure

```text
PATHFINDER_V26/
├── src/
│   ├── main.py                  # IMU trajectory estimator (writes TUM-format poses)
│   └── validate_output.py       # Validates generated trajectory output
├── datasets/
│   └── swirl_15s/
│       ├── DATASET_README.txt   # Dataset description
│       └── mav0/
│           ├── imu0/data.csv                          # 15 s synthetic IMU input
│           └── state_groundtruth_estimate0/data.csv   # Reference trajectory (dashboard only)
├── dashboard.py                 # Three-view trajectory playback dashboard
├── RUN_DEMO.bat                 # One-click demo launcher
├── metadata.json                # Run diagnostics written by the estimator
├── requirements.txt             # Python dependencies
└── PATHFINDER_V26_Presentation.pptx   # Final 9-slide presentation
```

---

## Quick start

### 1. Clone the repository

```powershell
git clone https://github.com/AravKotwal18/PATHFINDER_V26.git
cd PATHFINDER_V26
```

### 2. Install dependencies

```powershell
py -m pip install -r requirements.txt
```

### 3. Run the estimator

```powershell
py src/main.py "datasets/swirl_15s/mav0/imu0/data.csv" -o estimated_trajectory.txt
```

### 4. Validate the output

```powershell
py src/validate_output.py "datasets/swirl_15s/mav0/imu0/data.csv" estimated_trajectory.txt
```

### 5. Play the dashboard

```powershell
py dashboard.py estimated_trajectory.txt --ground-truth "datasets/swirl_15s/mav0/state_groundtruth_estimate0/data.csv"
```

Close the dashboard window to exit.

> **One-click option:** double-click `RUN_DEMO.bat` to run steps 2–5 in sequence. It stops if any step fails.

---

## Output format

The estimator writes poses in **TUM trajectory format**, one pose per input IMU sample (timestamps in seconds, quaternion order `qx qy qz qw`):

```text
# timestamp tx ty tz qx qy qz qw
```

The validator checks that the file has 8 columns per row, one pose per IMU sample, aligned timestamps, finite values, and unit-norm quaternions. It prints `PASS` or `FAIL` with the reason.

Each run also writes diagnostics to `metadata.json` (sample count, timestamp gaps, runtime, and so on), which overwrites the committed copy.

---

## Dataset

| Property | Value |
|---|---|
| Sequence | `swirl_15s` |
| Duration | 15 seconds |
| Sample rate | 200 Hz |
| IMU samples | 3,001 |
| Type | Synthetic, EuRoC-style stress-test data |

The reference trajectory is provided for offline visual comparison in the dashboard only. See `datasets/swirl_15s/DATASET_README.txt` for details.

---

## Design notes

- **`src/main.py`** is the estimator. It is offline and deterministic, and reads IMU data only (EuRoC `imu0/data.csv`, or a ZIP containing it).
- **`dashboard.py`** is a separate local viewer. It displays the estimator's output and does not generate or alter the trajectory.
- The dashboard is not part of the estimator's runtime or memory budget.

---

## Scope and limitations

- The included sequence is **synthetic**, not an official EuRoC recording.
- The reference trajectory **must not** be passed to the estimator.
- **No official ATE/RPE benchmark score is claimed.**
- Pure IMU position and yaw estimates can **drift** over time.
