# Pill Detection ROS Package

This ROS package implements a pill detection and tracking system that can operate in two modes:
1. Real-time detection using sensor hardware
2. Simulation using pre-recorded data for algorithm testing

## Overview

The system detects when a pill has been retrieved from a container by monitoring several sensor inputs:
- IMU sensor (accelerometer and orientation)
- Pressure sensor
- Light sensor

It can detect:
- Pick-place events (when the container is picked up and placed down)
- Pill retrieval events (using tilt, pressure, and light data)
- Swallow confirmation (through pressure holding)
- Possible forgotten pills (if swallow confirmation isn't received)

## Requirements

- ROS Noetic
- Python 3.x
- numpy
- matplotlib
- pandas

## Installation



1. Build the package:
   ```bash
   cd ~/catkin_ws
   catkin build
   ```

4. Source your workspace:
   ```bash
   source ~/catkin_ws/devel/setup.bash
   ```

## Data Format

### IMU Data Format
The package expects IMU data in CSV format with 6 columns:
```
ax,ay,az,r,p,y
```
Where:
- ax, ay, az are accelerometer readings in g (9.8 m/s²)
- r, p, y are orientation in degrees (roll, pitch, yaw)

### Pressure/Light Data Format
The package expects pressure and light data in space-separated format:
```
time pressure light unit
```
Where:
- time is in milliseconds
- pressure is a sensor value
- light is in lux
- unit is a text field (typically ignored)

## Usage

### Simulation Mode (Testing with Recorded Data)

1. Place your recorded data files in the data directory:
   ```bash
   cp your_imu_data.txt ~/catkin_ws/src/pill_detection/data/data_IMU.txt
   cp your_pressure_light_data.txt ~/catkin_ws/src/pill_detection/data/data_press_light.txt
   ```

2. Launch the simulation:
   ```bash
   roslaunch pill_detection pill_detection_simulation.launch
   ```

3. To visualize the data with custom plotting:
   ```bash
   roslaunch pill_detection pill_detection_visualization.launch
   ```

4. To run the full system with both simulation and visualization:
   ```bash
   roslaunch pill_detection pill_detection_full_system.launch
   ```

### Hardware Mode (Using Real Sensors)

1. Configure your hardware settings:
   ```bash
   rosparam set /pill_detection_node/imu_port "/dev/ttyUSB0"  # Change to match your setup
   rosparam set /pill_detection_node/pl_port "/dev/ttyUSB1"   # Change to match your setup
   ```

2. Launch the hardware mode:
   ```bash
   roslaunch pill_detection pill_detection_hardware.launch
   ```

## ROS Topics

The package publishes the following topics:

- `/pill_detection/imu`: IMU data (sensor_msgs/Imu)
- `/pill_detection/pressure_light`: Pressure and light data (pill_detection/PressureLight)
- `/pill_detection/accel_magnitude`: Acceleration magnitude and smoothing (geometry_msgs/Vector3)
- `/pill_detection/tilt_magnitude`: Tilt magnitude and change rate (geometry_msgs/Vector3)
- `/pill_detection/pick_place_events`: Pick-place events (pill_detection/PickPlaceEvent)
- `/pill_detection/pill_events`: Pill retrieval events (pill_detection/PillEvent)
- `/pill_detection/swallow_events`: Swallow confirmation events (pill_detection/SwallowEvent)
- `/pill_detection/forget_events`: Forgotten pill events (pill_detection/ForgetEvent)

## Parameters

The following parameters can be adjusted:

### Simulation Node
- `imu_file`: Path to IMU data file
- `pl_file`: Path to pressure/light data file
- `log_dir`: Directory for event logs
- `playback_speed`: Speed multiplier for simulation (default: 1.0)

### Hardware Node
- `imu_port`: Serial port for IMU sensor (e.g., /dev/ttyUSB0)
- `imu_baud_rate`: Baud rate for IMU (default: 512000)
- `pl_port`: Serial port for pressure/light sensors (e.g., /dev/ttyUSB1)
- `pl_baud_rate`: Baud rate for pressure/light sensors (default: 9600)

### Visualization Node
- `max_time_window`: Maximum time window to display in seconds (default: 60)
- `buffer_size`: Maximum number of data points to keep (default: 5000)

## Advanced Usage

### Adjusting Thresholds

You can adjust the detection thresholds by modifying the parameters in the code:

```python
# Default thresholds
self.motion_threshold = 0.02   # Acceleration threshold for motion detection
self.tilt_threshold = 7        # Orientation change threshold for tilt events
self.pressure_threshold = 60   # Pressure threshold for press events
self.light_threshold = 20      # Light threshold for light events
```

### Data Visualization

The package provides multiple visualization options:
1. Custom matplotlib visualization with the dedicated visualization node
2. RQT Plot for real-time data plotting
3. RQT Graph for visualizing the ROS node structure

## License

This project is licensed under the MIT License - see the LICENSE file for details.
