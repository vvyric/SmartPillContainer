# CACOM
Repo for project of CACOM: A Smart Pill Container for Accurate Pill Retrieval Detection and Disease Control

Project description:
Smart Pill Container is a low-cost medication adherence monitoring system designed to detect and record pill-taking behavior more accurately than conventional reminder-based solutions. The project combines an Arduino-based embedded platform with a 9-axis IMU, light sensor, and pressure sensor to identify key actions such as bottle pickup, opening, tilting, grasping, and active swallow confirmation. A multi-stage event detection algorithm was developed to reduce false positives and distinguish between successful medication intake, forgotten doses, and incomplete attempts. The system also includes ROS-based communication for IoT integration and supports statistical visualization of adherence patterns, enabling both patients and healthcare providers to better monitor medication routines and identify risk periods for non-adherence.

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

