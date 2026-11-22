# PATHFINDER V26 — Final Submission Package

Team: Agastaya Jindal (Leader), Arav Kotwal, Aditya Ranjan, Yogesh Jain

This folder contains the final editable presentation and only the files needed
to run the estimator, validate its output, and play the included 15-second demo.

## Included files

- `PATHFINDER_V26_Presentation.pptx` — final 9-slide presentation.
- `src/main.py` — IMU trajectory estimator; writes TUM-format poses.
- `src/validate_output.py` — validates generated trajectory output.
- `dashboard.py` — three-view trajectory playback dashboard.
- `requirements.txt` — Python dependencies.
- `datasets/swirl_15s/mav0/imu0/data.csv` — 15-second synthetic IMU input.
- `datasets/swirl_15s/mav0/state_groundtruth_estimate0/data.csv` — synthetic
  reference trajectory, used only by the dashboard for comparison.
- `datasets/swirl_15s/DATASET_README.txt` — dataset description.

## Run on Windows (PowerShell)

Open the extracted `PATHFINDER_Sentinel6D_Final` folder in VS Code. Open a
terminal in that folder and run:

```powershell
py -m pip install -r requirements.txt
py src/main.py "datasets/swirl_15s/mav0/imu0/data.csv" -o estimated_trajectory.txt
py src/validate_output.py "datasets/swirl_15s/mav0/imu0/data.csv" estimated_trajectory.txt
py dashboard.py estimated_trajectory.txt --ground-truth "datasets/swirl_15s/mav0/state_groundtruth_estimate0/data.csv"
```

The dashboard displays the estimator's output; it does not generate or alter
the trajectory. Close the dashboard window to exit.

## Important scope note

The included 15-second, 200 Hz sequence (3,001 IMU samples) is synthetic
EuRoC-style stress-test data, not an official EuRoC recording. Its reference
trajectory is for offline visual comparison only and must not be passed to the
estimator. No official ATE/RPE benchmark score is claimed. Pure IMU position
and yaw estimates can drift.

## Estimator / visualizer separation

`src/main.py` is the estimator. `dashboard.py` is a separate local demo viewer
and is not part of the estimator's runtime or memory budget.
