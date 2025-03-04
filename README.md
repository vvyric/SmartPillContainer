# CACOM
Repo for project of CACOM: A Smart Pill Container for Accurate Pill Retrieval Detection and Disease Control

## Folder Description:
- **hardware/**: Folder contains codes for hardware.
- **notebook/**: Folder contains basic analysis of raw data from hardware.
- **plots/**: Folder contains plots of data analysis.
- **log/**: Folder contains the logs the algorithm saves.
- **data/**: Folder contains the recorded raw data from the sensor.
- **scripts/**: Folder contains Python scripts:
  - `event_detection.py`: Normal event detection without swallow confirmation.
  - `event_detection_swallow_confirmation.py`: Enhanced event detection with swallow confirmation.
  - `generate_pill_log.py`: Script to generate synthetic data for statistical analysis.
  - `pill_taken_analyze.py`: Script to analyze the pill-taking log.
- **ros_ws/**: ROS package for the algorithm. Detailed description in `ros_ws/src/pill_detection/README.md`.

