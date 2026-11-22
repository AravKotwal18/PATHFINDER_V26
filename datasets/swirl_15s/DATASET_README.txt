15-second synthetic EuRoC-style IMU demonstration sequence, 200 Hz / 3,001 samples.
Includes a 0.55 s initialization interval followed by a rotating lateral acceleration
with periodic angular-rate modulation, generating several changing-radius swirls and
curved turns. Includes low sensor noise and six short vibration bursts.

Estimator input: mav0/imu0/data.csv
Reference only: mav0/state_groundtruth_estimate0/data.csv (offline evaluation).
Never pass ground truth to estimator. Synthetic, not official EuRoC data.
