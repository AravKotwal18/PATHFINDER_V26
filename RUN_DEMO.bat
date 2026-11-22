@echo off
cd /d "%~dp0"
py -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
py src\main.py "datasets\swirl_15s\mav0\imu0\data.csv" -o estimated_trajectory.txt
if errorlevel 1 exit /b 1
py src\validate_output.py "datasets\swirl_15s\mav0\imu0\data.csv" estimated_trajectory.txt
if errorlevel 1 exit /b 1
py dashboard.py estimated_trajectory.txt --ground-truth "datasets\swirl_15s\mav0\state_groundtruth_estimate0\data.csv"
