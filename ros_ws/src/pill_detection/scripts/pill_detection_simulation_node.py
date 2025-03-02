#!/usr/bin/env python3

import rospy
import numpy as np
import datetime
import time
import threading
from collections import deque
import pandas as pd
import os
import logging
from std_msgs.msg import Header
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3, Quaternion
from pill_detection.msg import PressureLight, PillEvent, SwallowEvent, ForgetEvent, PickPlaceEvent

class SensorDataBuffer:
    def __init__(self, max_size=1000):
        """Initialize data buffers for sensors with given max size."""
        self.max_size = max_size
        
        # IMU data
        self.imu_time = deque(maxlen=max_size)
        self.ax = deque(maxlen=max_size)
        self.ay = deque(maxlen=max_size)
        self.az = deque(maxlen=max_size)
        self.r = deque(maxlen=max_size)
        self.p = deque(maxlen=max_size)
        self.y = deque(maxlen=max_size)
        
        # Derived IMU data
        self.accel_magnitude = deque(maxlen=max_size)
        self.accel_deviation = deque(maxlen=max_size)
        self.accel_smooth = deque(maxlen=max_size)
        self.in_motion = deque(maxlen=max_size)
        self.state_change = deque(maxlen=max_size)
        self.tilt_magnitude = deque(maxlen=max_size)
        self.tilt_change = deque(maxlen=max_size)
        
        # Pressure/Light data
        self.pl_time = deque(maxlen=max_size)
        self.pressure = deque(maxlen=max_size)
        self.light = deque(maxlen=max_size)
        
        # Rest position (will be calibrated)
        self.rest_r = 0
        self.rest_p = 0
        self.rest_y = 80  # Default value, will be updated during calibration
        
        # Event tracking
        self.in_pick_place_event = False
        self.current_pick_time = None
        self.pick_place_events = []
        self.pill_events = []
        
        # Swallow confirmation tracking
        self.swallow_events = []
        self.forget_events = []
        self.awaiting_swallow = False
        self.pill_awaiting_swallow = None
        self.swallow_window_start = None
        self.swallow_window_duration = 10000  # 10 seconds in ms
        
        # Long press detection
        self.long_press_threshold = 1000  # 1 second in ms
        self.long_press_start = None
        self.in_long_press = False
        self.pressure_high_threshold = 100  # Higher threshold for long press
        
        # Initialize start time
        self.start_time = time.time() * 1000  # ms
        
        # Thresholds
        self.motion_threshold = 0.02
        self.tilt_threshold = 5
        self.pressure_threshold = 50
        self.light_threshold = 10
        self.min_event_duration = 1500  # ms
    
    def get_relative_time(self):
        """Get time in ms relative to start time"""
        return time.time() * 1000 - self.start_time
    
    def add_imu_reading(self, ax, ay, az, r, p, y):
        """Add a new IMU sensor reading to the buffer and update derived metrics"""
        current_time = self.get_relative_time()
        
        # Add raw readings
        self.imu_time.append(current_time)
        self.ax.append(ax)
        self.ay.append(ay)
        self.az.append(az)
        self.r.append(r)
        self.p.append(p)
        self.y.append(y)
        
        # Calculate derived values
        self._update_acceleration_metrics()
        self._update_tilt_metrics()
        
        # Check for swallow window expiration
        self._check_swallow_window_expiration(current_time)
    
    def add_pl_reading(self, pressure, light):
        """Add a new pressure/light sensor reading to the buffer"""
        current_time = self.get_relative_time()
        
        self.pl_time.append(current_time)
        self.pressure.append(pressure)
        self.light.append(light)
        
        # Check for long press (swallow confirmation)
        self._check_for_long_press(current_time, pressure)
        
        # Check for swallow window expiration
        self._check_swallow_window_expiration(current_time)
    
    # All of the following methods remain the same as in the original SensorDataBuffer class
    def _check_for_long_press(self, current_time, pressure):
        """Check for long press on pressure sensor (swallow confirmation)"""
        if self.awaiting_swallow:
            if pressure > self.pressure_high_threshold:
                if not self.in_long_press:
                    self.in_long_press = True
                    self.long_press_start = current_time
                    rospy.loginfo(f"Potential swallow confirmation started at {current_time/1000:.1f}s")
                elif current_time - self.long_press_start >= self.long_press_threshold:
                    # Long press detected - swallow confirmed
                    self._register_swallow_event(current_time)
            else:
                # Pressure released before threshold reached
                self.in_long_press = False
    
    def _register_swallow_event(self, current_time):
        """Register a confirmed swallow event"""
        if self.pill_awaiting_swallow is None:
            return
        
        swallow_event = {
            'pill_event': self.pill_awaiting_swallow,
            'swallow_time': current_time,
            'confirmation_delay': current_time - self.pill_awaiting_swallow['Time (ms)'],
            'actual_time': datetime.datetime.now()
        }
        
        self.swallow_events.append(swallow_event)
        
        seconds = current_time / 1000
        minutes = int(seconds // 60)
        seconds = seconds % 60
        delay = swallow_event['confirmation_delay'] / 1000
        
        log_message = (
            f"PILL SWALLOW CONFIRMED: Time {minutes}m {seconds:.1f}s\n"
            f"  - Pill taken at: {self.pill_awaiting_swallow['Time (ms)']/1000:.1f}s\n"
            f"  - Confirmation delay: {delay:.1f}s"
        )
        
        rospy.loginfo(f"\n✓✓✓ SWALLOW CONFIRMED ✓✓✓\n{log_message}")
        logging.info(f"SWALLOW CONFIRMED\n{log_message}")
        logging.info(f"Actual time of swallow confirmation: {swallow_event['actual_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Reset swallow tracking
        self.awaiting_swallow = False
        self.pill_awaiting_swallow = None
        self.swallow_window_start = None
        self.in_long_press = False
    
    def _check_swallow_window_expiration(self, current_time):
        """Check if the swallow confirmation window has expired"""
        if self.awaiting_swallow and self.swallow_window_start is not None:
            elapsed = current_time - self.swallow_window_start
            
            if elapsed >= self.swallow_window_duration:
                self._register_forget_event(current_time)
    
    def _register_forget_event(self, current_time):
        """Register a possible forget event when swallow window expires"""
        if self.pill_awaiting_swallow is None:
            return
        
        forget_event = {
            'pill_event': self.pill_awaiting_swallow,
            'forget_time': current_time,
            'window_duration': self.swallow_window_duration,
            'actual_time': datetime.datetime.now()
        }
        
        self.forget_events.append(forget_event)
        
        seconds = current_time / 1000
        minutes = int(seconds // 60)
        seconds = seconds % 60
        
        log_message = (
            f"POSSIBLE FORGOTTEN PILL: Time {minutes}m {seconds:.1f}s\n"
            f"  - Pill taken at: {self.pill_awaiting_swallow['Time (ms)']/1000:.1f}s\n"
            f"  - No swallow confirmation received within {self.swallow_window_duration/1000:.1f}s window"
        )
        
        rospy.loginfo(f"\n!!! POSSIBLE FORGOTTEN PILL !!!\n{log_message}")
        logging.info(f"POSSIBLE FORGOTTEN PILL\n{log_message}")
        logging.info(f"Actual time of forget event: {forget_event['actual_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Reset swallow tracking
        self.awaiting_swallow = False
        self.pill_awaiting_swallow = None
        self.swallow_window_start = None
        self.in_long_press = False
    
    def _update_acceleration_metrics(self):
        """Update derived acceleration metrics based on latest data with improved gravity handling"""
        if not self.ax:
            return
        
        # Calculate current acceleration magnitude
        mag = np.sqrt(self.ax[-1]**2 + self.ay[-1]**2 + self.az[-1]**2)
        self.accel_magnitude.append(mag)
        
        # Get rest acceleration if we have enough data
        if not hasattr(self, 'rest_ax') or not hasattr(self, 'rest_ay') or not hasattr(self, 'rest_az'):
            # Initialize rest acceleration with initial values or calibrated values
            if len(self.ax) >= 20:
                self.rest_ax = np.median(list(self.ax)[:20])
                self.rest_ay = np.median(list(self.ay)[:20])
                self.rest_az = np.median(list(self.az)[:20])
                rospy.loginfo(f"Calibrated rest acceleration: ax={self.rest_ax:.3f}, ay={self.rest_ay:.3f}, az={self.rest_az:.3f}")
            else:
                self.rest_ax = self.ax[0] if self.ax else 0
                self.rest_ay = self.ay[0] if self.ay else 0
                self.rest_az = self.az[0] if self.az else 0
        
        # Calculate dynamic acceleration by removing gravity component
        ax_dynamic = self.ax[-1] - self.rest_ax
        ay_dynamic = self.ay[-1] - self.rest_ay
        az_dynamic = self.az[-1] - self.rest_az
        
        # Calculate dynamic acceleration magnitude (without gravity)
        dynamic_mag = np.sqrt(ax_dynamic**2 + ay_dynamic**2 + az_dynamic**2)
        
        # Use the dynamic magnitude for motion detection
        if len(self.accel_deviation) >= 5:
            # Apply smoothing using a sliding window
            self.accel_deviation.append(dynamic_mag)
            window = list(self.accel_deviation)[-5:]
            smooth_val = np.mean(window)
            self.accel_smooth.append(smooth_val)
        else:
            self.accel_deviation.append(dynamic_mag)
            self.accel_smooth.append(dynamic_mag)
        
        # Log values periodically for debugging
        if len(self.ax) % 50 == 0:  # Log every 50 readings
            rospy.loginfo(f"Raw accel: ax={self.ax[-1]:.3f}, ay={self.ay[-1]:.3f}, az={self.az[-1]:.3f}")
            rospy.loginfo(f"Dynamic accel: ax={ax_dynamic:.3f}, ay={ay_dynamic:.3f}, az={az_dynamic:.3f}")
            rospy.loginfo(f"Dynamic magnitude: {dynamic_mag:.3f}, Smoothed: {self.accel_smooth[-1]:.3f}")
            rospy.loginfo(f"motion_threshold: {self.motion_threshold:.5f}, accel_smooth: {self.accel_smooth[-1]:.5f}")
        
        # Determine if in motion based on the smoothed dynamic magnitude
        in_motion = self.accel_smooth[-1] > self.motion_threshold
        self.in_motion.append(in_motion)
        
        # Calculate state changes for pick-place detection
        if len(self.in_motion) > 1:
            prev_state = 1 if self.in_motion[-2] else 0
            curr_state = 1 if self.in_motion[-1] else 0
            state_change = curr_state - prev_state
        else:
            state_change = 0
        
        self.state_change.append(state_change)
        self._update_pick_place_tracking()
    
    def _update_tilt_metrics(self):
        """Update tilt metrics based on latest orientation data with improved detection"""
        if not self.r:
            return
        
        # Initialize rest values if not already done
        if not hasattr(self, 'rest_r') or not hasattr(self, 'rest_p') or not hasattr(self, 'rest_y'):
            if len(self.r) >= 20:
                self.rest_r = np.median(list(self.r)[:20])
                self.rest_p = np.median(list(self.p)[:20])
                self.rest_y = np.median(list(self.y)[:20])
                rospy.loginfo(f"Calibrated rest orientation: r={self.rest_r:.2f}, p={self.rest_p:.2f}, y={self.rest_y:.2f}")
            else:
                self.rest_r = self.r[0] if self.r else 0
                self.rest_p = self.p[0] if self.p else 0
                self.rest_y = self.y[0] if self.y else 0
        
        # Calculate tilt differences from rest position with weights
        # Make yaw less sensitive since it often has larger natural variation
        r_weight, p_weight, y_weight = 1.0, 1.0, 0.3
        
        r_diff = r_weight * (self.r[-1] - self.rest_r)
        p_diff = p_weight * (self.p[-1] - self.rest_p)
        y_diff = y_weight * (self.y[-1] - self.rest_y)
        
        # Calculate tilt magnitude (3D angular distance from rest)
        tilt_mag = np.sqrt(r_diff**2 + p_diff**2 + y_diff**2)
        self.tilt_magnitude.append(tilt_mag)
        
        # Calculate tilt change rate (derivative)
        if len(self.tilt_magnitude) > 1:
            # Calculate change since previous reading
            tilt_change = abs(self.tilt_magnitude[-1] - self.tilt_magnitude[-2])
            
            # For more sensitivity, calculate change over last few readings
            if len(self.tilt_magnitude) >= 5:
                window = list(self.tilt_magnitude)[-5:]
                max_diff = max(window) - min(window)
                tilt_change = max(tilt_change, max_diff / 5)  # Use the greater of instantaneous and window change
        else:
            tilt_change = 0
        
        self.tilt_change.append(tilt_change)
        
        # Log values periodically for debugging
        if len(self.r) % 50 == 0:  # Log every 50 readings
            rospy.loginfo(f"Orientation: r={self.r[-1]:.2f}, p={self.p[-1]:.2f}, y={self.y[-1]:.2f}")
            rospy.loginfo(f"Rest: r={self.rest_r:.2f}, p={self.rest_p:.2f}, y={self.rest_y:.2f}")
            rospy.loginfo(f"Tilt magnitude: {tilt_mag:.2f}, Tilt change: {tilt_change:.2f}")
            rospy.loginfo(f"tilt_threshold: {self.tilt_threshold:.2f}")
        
        # Detect significant orientation changes directly
        if tilt_change > self.tilt_threshold:
            rospy.loginfo(f"Significant tilt change detected: {tilt_change:.2f} degrees")
    
    def _update_pick_place_tracking(self):
        """Update pick-place event tracking using both acceleration and orientation"""
        if not self.state_change or not self.tilt_change:
            return
        
        current_time = self.imu_time[-1]
        last_accel_change = self.state_change[-1]
        
        # Detect significant tilt change
        significant_tilt = False
        if len(self.tilt_change) > 0:
            significant_tilt = self.tilt_change[-1] > self.tilt_threshold
        
        # Detect pick-up either by acceleration change OR significant tilt
        pick_up_detected = (last_accel_change == 1) or (significant_tilt and not self.in_pick_place_event)
        
        if pick_up_detected and not self.in_pick_place_event:
            self.in_pick_place_event = True
            self.current_pick_time = current_time
            trigger = "acceleration" if last_accel_change == 1 else "tilt"
            rospy.loginfo(f"Pick up detected at {current_time/1000:.1f}s (trigger: {trigger})")
        
        # Place-down detection - use either acceleration or timeout
        place_down_detected = last_accel_change == -1
        
        # Add timeout detection - if 5+ seconds have passed since pick-up
        timeout_detected = False
        if self.in_pick_place_event and self.current_pick_time is not None:
            elapsed = current_time - self.current_pick_time
            if elapsed > 5000:  # 5 seconds timeout
                timeout_detected = True
                rospy.loginfo(f"Place down detected (timeout) at {current_time/1000:.1f}s")
        
        if (place_down_detected or timeout_detected) and self.in_pick_place_event:
            if self.current_pick_time is not None:
                duration = current_time - self.current_pick_time
                
                if duration >= self.min_event_duration:
                    event = {
                        'pick_time': self.current_pick_time,
                        'place_time': current_time,
                        'duration': duration
                    }
                    self.pick_place_events.append(event)
                    trigger = "acceleration" if place_down_detected else "timeout"
                    rospy.loginfo(f"Place down detected at {current_time/1000:.1f}s (duration: {duration/1000:.1f}s, trigger: {trigger})")
                    
                    self._check_for_pill_event(event)
            
            self.in_pick_place_event = False
            self.current_pick_time = None
    
    def _check_for_pill_event(self, pick_place_event):
        """Check if a pill retrieval event occurred within the pick-place window."""
        pick_time = pick_place_event['pick_time']
        place_time = pick_place_event['place_time']
        
        window_tilt_events = []
        for i in range(len(self.imu_time)):
            # rospy.loginfo(f"i: {i}, len(tilt_change): {len(self.tilt_change)}") 
            rospy.loginfo(f"tilt_change: {self.tilt_change}, tilt_threshold: {self.tilt_threshold}")
            time_val = self.imu_time[i]
            if pick_time <= time_val <= place_time:
                # rospy.loginfo(f"tilt_change: {self.tilt_change[i]}, tilt_threshold: {self.tilt_threshold}")
                if i < len(self.tilt_change) and self.tilt_change[i] > self.tilt_threshold:
                    window_tilt_events.append({
                        'time': time_val,
                        'tilt_change': self.tilt_change[i],
                        'r': self.r[i],
                        'p': self.p[i],
                        'y': self.y[i]
                    })
        
        window_pressure_events = []
        for i in range(len(self.pl_time)):
            time_val = self.pl_time[i]
            if pick_time <= time_val <= place_time:
                # rospy.loginfo(f"pressure: {self.pressure[i]}, pressure_threshold: {self.pressure_threshold}")
                if i < len(self.pressure) and self.pressure[i] > self.pressure_threshold:
                    window_pressure_events.append({
                        'time': time_val,
                        'pressure': self.pressure[i]
                    })
        
        window_light_events = []
        for i in range(len(self.pl_time)):
            time_val = self.pl_time[i]
            if pick_time <= time_val <= place_time:
                # rospy.loginfo(f"light: {self.light[i]}, light_threshold: {self.light_threshold}")
                if i < len(self.light) and self.light[i] > self.light_threshold:
                    window_light_events.append({
                        'time': time_val,
                        'light': self.light[i]
                    })
        
        if window_tilt_events and window_pressure_events and window_light_events:
            max_tilt_event = max(window_tilt_events, key=lambda x: x['tilt_change'])
            max_pressure_event = max(window_pressure_events, key=lambda x: x['pressure'])
            max_light_event = max(window_light_events, key=lambda x: x['light'])
            
            pill_event = {
                'Time (ms)': max_tilt_event['time'],
                'Tilt': max_tilt_event['tilt_change'],
                'r': max_tilt_event['r'],
                'p': max_tilt_event['p'],
                'y': max_tilt_event['y'],
                'Pressure': max_pressure_event['pressure'],
                'Light': max_light_event['light'],
                'Pressure_time': max_pressure_event['time'],
                'Light_time': max_light_event['time'],
                'pick_time': pick_time,
                'place_time': place_time,
                'actual_time': datetime.datetime.now(),
                'confirmed': False,  # Track if pill has been confirmed swallowed
                'forgotten': False   # Track if pill might have been forgotten
            }
            
            self.pill_events.append(pill_event)
            seconds = pill_event['Time (ms)'] / 1000
            minutes = int(seconds // 60)
            seconds = seconds % 60
            
            log_message = (
                f"Pill Event: Time {minutes}m {seconds:.1f}s\n"
                f"  - Within Pick-Place window: {pick_time/1000:.1f}s to {place_time/1000:.1f}s\n"
                f"  - 3D Tilt: {pill_event['Tilt']:.2f}\n"
                f"  - Orientation: Roll={pill_event['r']:.2f}°, Pitch={pill_event['p']:.2f}°, Yaw={pill_event['y']:.2f}°\n"
                f"  - Pressure: {pill_event['Pressure']:.0f} at {pill_event['Pressure_time']/1000:.1f}s\n"
                f"  - Light: {pill_event['Light']:.2f} lux at {pill_event['Light_time']/1000:.1f}s\n"
                f"  - Awaiting swallow confirmation (press and hold pressure sensor within 10s)"
            )
            
            rospy.loginfo(f"\n!!! PILL EVENT DETECTED !!!\n{log_message}")
            logging.info(f"PILL EVENT DETECTED\n{log_message}")
            logging.info(f"Actual time of event: {pill_event['actual_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Start swallow confirmation window
            self.awaiting_swallow = True
            self.pill_awaiting_swallow = pill_event
            self.swallow_window_start = pill_event['Time (ms)']
    
    def calibrate_rest_position(self, num_samples=20):
        """Calibrate the rest position of the device using initial readings"""
        if len(self.r) >= num_samples and len(self.p) >= num_samples and len(self.y) >= num_samples:
            self.rest_r = np.median(list(self.r)[:num_samples])
            self.rest_p = np.median(list(self.p)[:num_samples])
            self.rest_y = np.median(list(self.y)[:num_samples])
            
            rospy.loginfo(f"Calibrated rest position: r={self.rest_r:.2f}, p={self.rest_p:.2f}, y={self.rest_y:.2f}")
            return True
        return False


class PillDetectionSimulationNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('pill_detection_simulation_node', anonymous=True)
        
        # Get parameters from ROS parameter server
        self.imu_file = rospy.get_param('~imu_file', 'data_IMU.txt')
        self.pl_file = rospy.get_param('~pl_file', 'data_press_light.txt')
        self.log_dir = rospy.get_param('~log_dir', 'pill_logs')
        self.playback_speed = rospy.get_param('~playback_speed', 1.0)
        self.force_events = rospy.get_param('~force_events', False)
        self.use_synthetic_data = rospy.get_param('~use_synthetic_data', False)
        self.debug_level = rospy.get_param('~debug_level', 'info')
        
        # Set up logging
        self._setup_logging()
        
        # Initialize data buffer
        self.sensor_buffer = SensorDataBuffer(max_size=1000)
        
        # Set up ROS publishers
        self._setup_publishers()
        
        # Load recorded or synthetic data
        if self.use_synthetic_data:
            rospy.loginfo("Using synthetic test data")
            self.imu_data, self.pl_data = self._create_synthetic_test_data()
            
            # Set appropriate thresholds for synthetic data
            self.sensor_buffer.motion_threshold = 0.05
            self.sensor_buffer.tilt_threshold = 5.0
            self.sensor_buffer.pressure_threshold = 50.0
            self.sensor_buffer.light_threshold = 50.0
        else:
            rospy.loginfo("Using real data files")
            self.imu_data, self.pl_data = self._load_recorded_data()
        
        # Override thresholds with parameters if provided
        self.sensor_buffer.motion_threshold = rospy.get_param('~motion_threshold', self.sensor_buffer.motion_threshold)
        self.sensor_buffer.tilt_threshold = rospy.get_param('~tilt_threshold', self.sensor_buffer.tilt_threshold)
        self.sensor_buffer.pressure_threshold = rospy.get_param('~pressure_threshold', self.sensor_buffer.pressure_threshold)
        self.sensor_buffer.light_threshold = rospy.get_param('~light_threshold', self.sensor_buffer.light_threshold)
        
        rospy.loginfo(f"Using thresholds: motion={self.sensor_buffer.motion_threshold:.3f}, " +
                f"tilt={self.sensor_buffer.tilt_threshold:.1f}, " +
                f"pressure={self.sensor_buffer.pressure_threshold:.1f}, " +
                f"light={self.sensor_buffer.light_threshold:.1f}")
        
        # Flag for node status
        self.running = True
        
        # Simulation time tracking (ms)
        self.sim_time = 0
        self.sim_start_time = None
        
        # Data indices for playback
        self.imu_index = 0
        self.pl_index = 0
        
        # Log data loading to file
        self._log_to_file(f"Data loaded successfully: {len(self.imu_data)} IMU points, {len(self.pl_data)} pressure/light points")
        self._log_to_file(f"Thresholds: motion={self.sensor_buffer.motion_threshold:.3f}, " +
                    f"tilt={self.sensor_buffer.tilt_threshold:.1f}, " +
                    f"pressure={self.sensor_buffer.pressure_threshold:.1f}, " +
                    f"light={self.sensor_buffer.light_threshold:.1f}")
        
        rospy.loginfo("Pill detection simulation node initialized")
    
    def _setup_logging(self):
        """Set up logging for pill events with robust error handling"""
        try:
            # Create absolute path for logging
            log_dir = os.path.abspath(self.log_dir)
            os.makedirs(log_dir, exist_ok=True)
            
            log_filename = os.path.join(
                log_dir, 
                f"pill_events_sim_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            
            # Test if we can write to this location
            try:
                with open(log_filename, 'w') as test_log:
                    test_log.write("Log initialized\n")
                    test_log.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Configure logging
                logging.basicConfig(
                    filename=log_filename, 
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S',
                    force=True  # Force reconfiguration in case logging was already set up
                )
                
                # Add a file handler to make sure logs are written
                file_handler = logging.FileHandler(log_filename, mode='a')
                file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', '%Y-%m-%d %H:%M:%S'))
                
                # Get the root logger and add our file handler
                root_logger = logging.getLogger()
                root_logger.handlers = []  # Remove existing handlers
                root_logger.addHandler(file_handler)
                root_logger.setLevel(logging.INFO)
                
                self.log_filename = log_filename
                rospy.loginfo(f"Successfully created and writing to log at: {log_filename}")
                
                # Write the first log entry
                rospy.loginfo("Logging system initialized")
                logging.info("LOGGING STARTED")
                logging.info(f"ROS node: {rospy.get_name()}")
                logging.info(f"IMU file: {self.imu_file}")
                logging.info(f"Pressure/Light file: {self.pl_file}")
                
            except (IOError, PermissionError) as e:
                # Try a fallback location
                self._setup_fallback_logging(f"Error writing to log file: {e}")
        
        except Exception as e:
            self._setup_fallback_logging(f"Error setting up logging: {e}")

    def _setup_fallback_logging(self, error_msg):
        """Set up logging in a fallback location if the primary location fails"""
        rospy.logerr(error_msg)
        
        try:
            # Try using /tmp as a fallback
            fallback_log = f"/tmp/pill_events_sim_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            rospy.logwarn(f"Falling back to temporary log location: {fallback_log}")
            
            logging.basicConfig(
                filename=fallback_log,
                level=logging.INFO,
                format='%(asctime)s - %(message)s', 
                datefmt='%Y-%m-%d %H:%M:%S',
                force=True
            )
            
            # Add a direct write to make sure something gets logged
            with open(fallback_log, 'w') as f:
                f.write(f"Log initialized at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"ERROR: {error_msg}\n")
            
            self.log_filename = fallback_log
            logging.info("LOGGING STARTED (FALLBACK)")
            logging.info(f"Original error: {error_msg}")
            
        except Exception as e:
            rospy.logerr(f"Failed to set up even fallback logging: {e}")
            self.log_filename = None

    # Helper function to log events explicitly to file
    def _log_to_file(self, message, level="INFO"):
        """Write a log message directly to the log file"""
        if self.log_filename:
            try:
                with open(self.log_filename, 'a') as f:
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f"{timestamp} - {level} - {message}\n")
            except Exception as e:
                rospy.logerr(f"Error writing to log file: {e}")
        
        # Also log to Python's logging system
        if level == "INFO":
            logging.info(message)
        elif level == "WARNING":
            logging.warning(message)
        elif level == "ERROR":
            logging.error(message)
    
    def _print_data_summary(self, imu_data, pl_data):
        """Print summary statistics of loaded data to help debug threshold settings"""
        try:
            # Print basic statistics
            rospy.loginfo("=== DATA SUMMARY ===")
            
            # IMU data
            ax_min, ax_max = imu_data['ax'].min(), imu_data['ax'].max()
            ay_min, ay_max = imu_data['ay'].min(), imu_data['ay'].max()
            az_min, az_max = imu_data['az'].min(), imu_data['az'].max()
            
            r_min, r_max = imu_data['r'].min(), imu_data['r'].max()
            p_min, p_max = imu_data['p'].min(), imu_data['p'].max()
            y_min, y_max = imu_data['y'].min(), imu_data['y'].max()
            
            rospy.loginfo(f"IMU Data Range:")
            rospy.loginfo(f"  Acceleration (g): ax=[{ax_min:.3f}, {ax_max:.3f}], ay=[{ay_min:.3f}, {ay_max:.3f}], az=[{az_min:.3f}, {az_max:.3f}]")
            rospy.loginfo(f"  Orientation (deg): r=[{r_min:.1f}, {r_max:.1f}], p=[{p_min:.1f}, {p_max:.1f}], y=[{y_min:.1f}, {y_max:.1f}]")
            
            # Pressure/Light data
            p_min, p_max = pl_data['Pressure'].min(), pl_data['Pressure'].max()
            l_min, l_max = pl_data['Light'].min(), pl_data['Light'].max()
            
            rospy.loginfo(f"Pressure/Light Data Range:")
            rospy.loginfo(f"  Pressure: [{p_min:.1f}, {p_max:.1f}]")
            rospy.loginfo(f"  Light: [{l_min:.1f}, {l_max:.1f}]")
            
            # Calculate suitable thresholds based on data
            accel_threshold = max(0.02, (ax_max - ax_min + ay_max - ay_min + az_max - az_min) / 30)
            tilt_threshold = max(5.0, (r_max - r_min + p_max - p_min) / 20)
            pressure_threshold = max(20.0, p_min + (p_max - p_min) * 0.25)  # 25% of range above min
            light_threshold = max(15.0, l_min + (l_max - l_min) * 0.25)  # 25% of range above min
            
            rospy.loginfo(f"Suggested Thresholds:")
            rospy.loginfo(f"  Motion threshold: {accel_threshold:.3f}")
            rospy.loginfo(f"  Tilt threshold: {tilt_threshold:.1f}")
            rospy.loginfo(f"  Pressure threshold: {pressure_threshold:.1f}")
            rospy.loginfo(f"  Light threshold: {light_threshold:.1f}")
            
            # Return the suggested thresholds
            return {
                'motion_threshold': accel_threshold,
                'tilt_threshold': tilt_threshold,
                'pressure_threshold': pressure_threshold,
                'light_threshold': light_threshold
            }
        except Exception as e:
            rospy.logerr(f"Error analyzing data: {e}")
            return {
                'motion_threshold': 0.03,
                'tilt_threshold': 5.0,
                'pressure_threshold': 50.0,
                'light_threshold': 15.0
            }

    def _load_recorded_data(self):
        try:
            # IMU data loading
            rospy.loginfo(f"Loading IMU data from: {self.imu_file}")
            imu_data = self._load_imu_data(self.imu_file)
            
            # Pressure/Light data loading
            rospy.loginfo(f"Loading pressure/light data from: {self.pl_file}")
            pl_data = self._load_pressure_light_data(self.pl_file)
            
            if imu_data is None or pl_data is None:
                rospy.logerr("Failed to load data")
                return None, None
                
            rospy.loginfo(f"Loaded {len(imu_data)} IMU data points and {len(pl_data)} pressure/light data points")
            
            # Print data summary and get suggested thresholds
            suggested_thresholds = self._print_data_summary(imu_data, pl_data)
            
            # Override threshold parameters if provided
            self.sensor_buffer.motion_threshold = rospy.get_param('~motion_threshold', suggested_thresholds['motion_threshold'])
            self.sensor_buffer.tilt_threshold = rospy.get_param('~tilt_threshold', suggested_thresholds['tilt_threshold'])
            self.sensor_buffer.pressure_threshold = rospy.get_param('~pressure_threshold', suggested_thresholds['pressure_threshold'])
            self.sensor_buffer.light_threshold = rospy.get_param('~light_threshold', suggested_thresholds['light_threshold'])
            
            rospy.loginfo(f"Using thresholds: motion={self.sensor_buffer.motion_threshold:.3f}, " +
                    f"tilt={self.sensor_buffer.tilt_threshold:.1f}, " +
                    f"pressure={self.sensor_buffer.pressure_threshold:.1f}, " +
                    f"light={self.sensor_buffer.light_threshold:.1f}")
                    
            return imu_data, pl_data
            
        except Exception as e:
            rospy.logerr(f"Error loading recorded data: {e}")
            rospy.signal_shutdown("Data loading failure")
            return None, None
    
    def _load_imu_data(self, file_path):
        """Load IMU data from file"""
        try:
            # Check if the file exists
            if not os.path.exists(file_path):
                rospy.logerr(f"IMU data file not found: {file_path}")
                return pd.DataFrame(columns=["ax", "ay", "az", "r", "p", "y", "Time (ms)"])
            
            # Read all IMU data with correct column names
            full_imu_df = pd.read_csv(file_path, header=None, 
                                  names=["ax", "ay", "az", "r", "p", "y"])
            
            # Skip the first 80 rows as specified in the reference code
            imu_df = full_imu_df.iloc[80:].reset_index(drop=True)
            
            # Add time column (0.5s intervals) starting from 0
            imu_df["Time (ms)"] = [i * 500 for i in range(len(imu_df))]
            
            return imu_df
        except Exception as e:
            rospy.logerr(f"Error loading IMU data: {e}")
            return pd.DataFrame(columns=["ax", "ay", "az", "r", "p", "y", "Time (ms)"])
    
    def _load_pressure_light_data(self, file_path):
        """Load pressure and light data from file"""
        try:
            # Check if the file exists
            if not os.path.exists(file_path):
                rospy.logerr(f"Pressure/Light data file not found: {file_path}")
                return pd.DataFrame(columns=["Time (ms)", "Pressure", "Light"])
                
            # Read all pressure and light data
            full_press_light_df = pd.read_csv(file_path, delim_whitespace=True, header=None, 
                                      names=["Time (ms)", "Pressure", "Light Intensity (lux)", "Unit"])
            
            # Skip the first 139 rows as specified in the reference code
            press_light_df = full_press_light_df.iloc[139:].reset_index(drop=True)
            
            # Drop the unnecessary unit column
            press_light_df = press_light_df.drop(columns=["Unit"])
            
            # Make sure we have data after filtering
            if press_light_df.empty:
                rospy.logwarn("No pressure/light data available after filtering")
                return pd.DataFrame(columns=["Time (ms)", "Pressure", "Light"])
                
            # Adjust timestamps to start from 0
            first_timestamp = press_light_df["Time (ms)"].iloc[0]
            press_light_df["Time (ms)"] = press_light_df["Time (ms)"] - first_timestamp
            
            # Rename column to match our code
            press_light_df = press_light_df.rename(columns={"Light Intensity (lux)": "Light"})
            
            return press_light_df
        except Exception as e:
            rospy.logerr(f"Error loading pressure/light data: {e}")
            return pd.DataFrame(columns=["Time (ms)", "Pressure", "Light"])
        
    def _setup_publishers(self):
        """Set up ROS publishers for sensor data and events"""
        # Raw data publishers
        self.imu_pub = rospy.Publisher('pill_detection/imu', Imu, queue_size=10)
        self.pressure_light_pub = rospy.Publisher('pill_detection/pressure_light', PressureLight, queue_size=10)
        
        # Derived metrics publishers
        self.accel_mag_pub = rospy.Publisher('pill_detection/accel_magnitude', Vector3, queue_size=10)
        self.tilt_mag_pub = rospy.Publisher('pill_detection/tilt_magnitude', Vector3, queue_size=10)
        
        # Event publishers
        self.pick_place_pub = rospy.Publisher('pill_detection/pick_place_events', PickPlaceEvent, queue_size=10)
        self.pill_event_pub = rospy.Publisher('pill_detection/pill_events', PillEvent, queue_size=10)
        self.swallow_event_pub = rospy.Publisher('pill_detection/swallow_events', SwallowEvent, queue_size=10)
        self.forget_event_pub = rospy.Publisher('pill_detection/forget_events', ForgetEvent, queue_size=10)
        
        # DEBUG publishers for better visualization with rqt_plot
        # Thresholds as separate topics
        self.thresholds_pub = rospy.Publisher('pill_detection/thresholds', Vector3, queue_size=10) 
        
        # Motion state for detecting pick-place events
        self.motion_state_pub = rospy.Publisher('pill_detection/motion_state', Vector3, queue_size=10)
        
        # Event detection status
        self.event_status_pub = rospy.Publisher('pill_detection/event_status', Vector3, queue_size=10)
        
        # DEBUG publishers for better visualization with rqt_plot
        # Thresholds as separate topics
        self.thresholds_pub = rospy.Publisher('pill_detection/thresholds', Vector3, queue_size=10) 
        
        # Motion state for detecting pick-place events
        self.motion_state_pub = rospy.Publisher('pill_detection/motion_state', Vector3, queue_size=10)
        
        # Event detection status
        self.event_status_pub = rospy.Publisher('pill_detection/event_status', Vector3, queue_size=10)
        
        
    # Add this function to generate synthetic data that will definitely trigger events
    def _create_synthetic_test_data(self):
        """Create synthetic test data that will trigger events for testing"""
        rospy.loginfo("Creating synthetic test data")
        
        # Create synthetic IMU data
        imu_data = pd.DataFrame({
            'ax': np.zeros(200),
            'ay': np.zeros(200),
            'az': np.ones(200),
            'r': np.zeros(200),
            'p': np.zeros(200),
            'y': np.zeros(200),
            'Time (ms)': [i * 500 for i in range(200)]
        })
        
        # Create synthetic pressure/light data
        pl_data = pd.DataFrame({
            'Time (ms)': [i * 500 for i in range(200)],
            'Pressure': np.zeros(200),
            'Light': np.zeros(200)
        })
        
        # Add motion events at specific times
        # Event 1: t=10s to t=15s
        for i in range(20, 30):
            imu_data.loc[i, 'ax'] = 0.5  # Add some acceleration to trigger motion
        
        # Event 2: t=30s to t=35s
        for i in range(60, 70):
            imu_data.loc[i, 'ax'] = 0.5  # Add some acceleration to trigger motion
        
        # Event 3: t=50s to t=55s
        for i in range(100, 110):
            imu_data.loc[i, 'ax'] = 0.5  # Add some acceleration to trigger motion
            
        # Add tilt changes within motion windows
        imu_data.loc[25, 'r'] = 50.0  # Add tilt in first window
        imu_data.loc[65, 'p'] = 50.0  # Add tilt in second window
        imu_data.loc[105, 'y'] = 50.0  # Add tilt in third window
        
        # Add pressure and light events within the same windows
        pl_data.loc[25, 'Pressure'] = 200.0  # Add pressure in first window
        pl_data.loc[25, 'Light'] = 200.0     # Add light in first window
        
        pl_data.loc[65, 'Pressure'] = 200.0  # Add pressure in second window
        pl_data.loc[65, 'Light'] = 200.0     # Add light in second window
        
        pl_data.loc[105, 'Pressure'] = 200.0  # Add pressure in third window
        pl_data.loc[105, 'Light'] = 200.0     # Add light in third window
        
        rospy.loginfo("Synthetic test data created with 3 events at t=10-15s, t=30-35s, and t=50-55s")
        
        return imu_data, pl_data
    
    def start(self):
        # Load parameters
        self.force_events = rospy.get_param('~force_events', False)
        """Start the simulation"""
        rospy.loginfo("Starting pill detection simulation...")
        
        # Initialize simulation time
        self.sim_start_time = time.time()
        self.sensor_buffer.start_time = 0  # Set start time to 0 for simulation
        
        # Calibrate sensors with initial data points
        self._calibrate_sensors()
        
        # Main simulation loop
        rate = rospy.Rate(50)  # 50 Hz update rate
        
        try:
            while self.running and not rospy.is_shutdown():
                # Update simulation time
                self.sim_time = (time.time() - self.sim_start_time) * 1000 * self.playback_speed
                
                # Feed data points that match the current simulation time
                self._feed_data_points()
                
                # Force pill events if enabled
                if self.force_events:
                    self._force_pill_event()
                
                # Publish current state to ROS topics
                self._publish_data()
                
                # Check if we've reached the end of the data
                if self.imu_index >= len(self.imu_data) and self.pl_index >= len(self.pl_data):
                    rospy.loginfo("Reached end of recorded data. Simulation complete.")
                    break
                
                rate.sleep()
        
        except KeyboardInterrupt:
            rospy.loginfo("Simulation stopped by user.")
        
        finally:
            self._print_summary()
            
        # Keep the node running to allow inspection of results
        rospy.spin()
    
    def _calibrate_sensors(self):
        """Calibrate sensors using initial data points"""
        rospy.loginfo("Calibrating sensors with initial data...")
        
        # Use the first 20 data points for calibration
        calibration_samples = min(20, len(self.imu_data))
        
        for i in range(calibration_samples):
            row = self.imu_data.iloc[i]
            self.sensor_buffer.add_imu_reading(
                row['ax'], row['ay'], row['az'], 
                row['r'], row['p'], row['y']
            )
        
        # Perform calibration
        calibrated = self.sensor_buffer.calibrate_rest_position()
        
        if calibrated:
            rospy.loginfo("Sensor calibration complete.")
        else:
            rospy.logwarn("Sensor calibration failed - using default values.")
    
    def _feed_data_points(self):
        """Feed data points based on current simulation time"""
        # Feed IMU data points that match the current simulation time
        points_added = 0
        while (self.imu_index < len(self.imu_data) and 
            self.imu_data.iloc[self.imu_index]['Time (ms)'] <= self.sim_time):
            row = self.imu_data.iloc[self.imu_index]
            self.sensor_buffer.add_imu_reading(
                row['ax'], row['ay'], row['az'], 
                row['r'], row['p'], row['y']
            )
            self.imu_index += 1
            points_added += 1
        
        # Feed pressure/light data points that match the current simulation time
        pl_points_added = 0
        while (self.pl_index < len(self.pl_data) and 
            self.pl_data.iloc[self.pl_index]['Time (ms)'] <= self.sim_time):
            row = self.pl_data.iloc[self.pl_index]
            self.sensor_buffer.add_pl_reading(
                row['Pressure'], row['Light']
            )
            self.pl_index += 1
            pl_points_added += 1
        
        # Log data progress occasionally (every 5 seconds)
        if hasattr(self, 'last_progress_time'):
            if self.sim_time - self.last_progress_time > 5000:  # 5 seconds
                progress_imu = (self.imu_index / len(self.imu_data)) * 100 if len(self.imu_data) > 0 else 0
                progress_pl = (self.pl_index / len(self.pl_data)) * 100 if len(self.pl_data) > 0 else 0
                
                rospy.loginfo(f"Progress: IMU {progress_imu:.1f}%, PL {progress_pl:.1f}% at t={self.sim_time/1000:.1f}s")
                self.last_progress_time = self.sim_time
        else:
            self.last_progress_time = self.sim_time
        
        # Log when events are detected in a clearer way
        if len(self.sensor_buffer.pick_place_events) > 0 and (not hasattr(self, 'last_reported_pp_count') or 
                                                        len(self.sensor_buffer.pick_place_events) > self.last_reported_pp_count):
            rospy.loginfo(f"PICK-PLACE EVENT DETECTED: {len(self.sensor_buffer.pick_place_events)} total")
            self.last_reported_pp_count = len(self.sensor_buffer.pick_place_events)
            
        if len(self.sensor_buffer.pill_events) > 0 and (not hasattr(self, 'last_reported_pill_count') or 
                                                len(self.sensor_buffer.pill_events) > self.last_reported_pill_count):
            new_event = self.sensor_buffer.pill_events[-1]
            time_s = new_event['Time (ms)'] / 1000
            rospy.loginfo(f"PILL EVENT DETECTED: {len(self.sensor_buffer.pill_events)} total at t={time_s:.1f}s")
            
            # Log detailed event info to file
            event_details = (
                f"PILL EVENT {len(self.sensor_buffer.pill_events)}:\n"
                f"  Time: {time_s:.1f}s\n"
                f"  Tilt: {new_event['Tilt']:.2f}\n"
                f"  Orientation: r={new_event['r']:.1f}, p={new_event['p']:.1f}, y={new_event['y']:.1f}\n"
                f"  Pressure: {new_event['Pressure']:.1f}\n"
                f"  Light: {new_event['Light']:.1f}\n"
                f"  Pick-Place: {new_event['pick_time']/1000:.1f}s - {new_event['place_time']/1000:.1f}s"
            )
            self._log_to_file(event_details)
            
            self.last_reported_pill_count = len(self.sensor_buffer.pill_events)
            
    def _publish_debug_data(self):
        """Publish debug data for easier visualization in rqt_plot"""
        # Publish current thresholds 
        threshold_msg = Vector3()
        threshold_msg.x = self.sensor_buffer.motion_threshold  # Motion threshold
        threshold_msg.y = self.sensor_buffer.tilt_threshold    # Tilt threshold
        threshold_msg.z = self.sensor_buffer.pressure_threshold # Pressure threshold
        self.thresholds_pub.publish(threshold_msg)
        
        # Publish motion state
        motion_msg = Vector3()
        if self.sensor_buffer.in_motion and len(self.sensor_buffer.in_motion) > 0:
            motion_msg.x = 1.0 if self.sensor_buffer.in_motion[-1] else 0.0  # Current motion state
        else:
            motion_msg.x = 0.0
            
        motion_msg.y = 1.0 if self.sensor_buffer.in_pick_place_event else 0.0  # Pick-place active
        
        if self.sensor_buffer.state_change and len(self.sensor_buffer.state_change) > 0:
            motion_msg.z = self.sensor_buffer.state_change[-1]  # State change (-1, 0, or 1)
        else:
            motion_msg.z = 0.0
        
        self.motion_state_pub.publish(motion_msg)
        
        # Publish event status
        event_msg = Vector3()
        event_msg.x = len(self.sensor_buffer.pick_place_events)  # Number of pick-place events
        event_msg.y = len(self.sensor_buffer.pill_events)       # Number of pill events
        event_msg.z = 1.0 if self.sensor_buffer.awaiting_swallow else 0.0  # Awaiting swallow
        self.event_status_pub.publish(event_msg)

    
    def _publish_data(self):
        """Publish sensor data and events to ROS topics"""
        # Create header with current timestamp
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "pill_sensor"
        
        # Publish IMU data if available
        if self.sensor_buffer.ax and self.sensor_buffer.ay and self.sensor_buffer.az:
            imu_msg = Imu()
            imu_msg.header = header
            
            # Linear acceleration
            imu_msg.linear_acceleration.x = self.sensor_buffer.ax[-1]
            imu_msg.linear_acceleration.y = self.sensor_buffer.ay[-1]
            imu_msg.linear_acceleration.z = self.sensor_buffer.az[-1]
            
            # We don't have angular velocity data, so leave it as zeros
            
            # For orientation, convert roll, pitch, yaw to quaternion
            # This is a simplified conversion - for proper conversion consider using tf transformations
            r, p, y = self.sensor_buffer.r[-1], self.sensor_buffer.p[-1], self.sensor_buffer.y[-1]
            # Convert to radians
            r_rad, p_rad, y_rad = np.radians(r), np.radians(p), np.radians(y)
            
            # Simple conversion to quaternion (this is an approximation)
            cy = np.cos(y_rad * 0.5)
            sy = np.sin(y_rad * 0.5)
            cp = np.cos(p_rad * 0.5)
            sp = np.sin(p_rad * 0.5)
            cr = np.cos(r_rad * 0.5)
            sr = np.sin(r_rad * 0.5)
            
            imu_msg.orientation.w = cy * cp * cr + sy * sp * sr
            imu_msg.orientation.x = cy * cp * sr - sy * sp * cr
            imu_msg.orientation.y = sy * cp * sr + cy * sp * cr
            imu_msg.orientation.z = sy * cp * cr - cy * sp * sr
            
            self.imu_pub.publish(imu_msg)
            
            # Publish derived acceleration magnitude
            accel_mag_msg = Vector3()
            accel_mag_msg.x = self.sensor_buffer.accel_magnitude[-1]
            accel_mag_msg.y = self.sensor_buffer.accel_smooth[-1] if self.sensor_buffer.accel_smooth else 0
            accel_mag_msg.z = 1.0 if self.sensor_buffer.in_motion and self.sensor_buffer.in_motion[-1] else 0.0
            self.accel_mag_pub.publish(accel_mag_msg)
            
            # Publish tilt magnitude
            tilt_mag_msg = Vector3()
            tilt_mag_msg.x = self.sensor_buffer.tilt_magnitude[-1] if self.sensor_buffer.tilt_magnitude else 0
            tilt_mag_msg.y = self.sensor_buffer.tilt_change[-1] if self.sensor_buffer.tilt_change else 0
            tilt_mag_msg.z = 0.0  # Unused
            self.tilt_mag_pub.publish(tilt_mag_msg)
        
        # Publish pressure/light data if available
        if self.sensor_buffer.pressure and self.sensor_buffer.light:
            pl_msg = PressureLight()
            pl_msg.header = header
            pl_msg.pressure = self.sensor_buffer.pressure[-1]
            pl_msg.light = self.sensor_buffer.light[-1]
            self.pressure_light_pub.publish(pl_msg)
        
        # Publish new events as they occur
        self._publish_new_events()
        # Publish debug data
        self._publish_debug_data()
        
    def _publish_new_events(self):
        """Publish any new events as they are detected"""
        # Used to track the events we've already published
        if not hasattr(self, 'last_pick_place_index'):
            self.last_pick_place_index = 0
            self.last_pill_event_index = 0
            self.last_swallow_event_index = 0
            self.last_forget_event_index = 0
        
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "pill_sensor"
        
        # Publish new pick-place events
        for i in range(self.last_pick_place_index, len(self.sensor_buffer.pick_place_events)):
            event = self.sensor_buffer.pick_place_events[i]
            pp_msg = PickPlaceEvent()
            pp_msg.header = header
            
            # Convert ms to ROS time - ensure positive values
            pick_secs = max(0, int(event['pick_time'] / 1000))
            pick_nsecs = max(0, int((event['pick_time'] % 1000) * 1e6))
            place_secs = max(0, int(event['place_time'] / 1000))
            place_nsecs = max(0, int((event['place_time'] % 1000) * 1e6))
            
            pp_msg.pick_time = rospy.Time(pick_secs, pick_nsecs)
            pp_msg.place_time = rospy.Time(place_secs, place_nsecs)
            pp_msg.duration = max(0, event['duration'] / 1000.0)  # Convert to seconds, ensure positive
            
            self.pick_place_pub.publish(pp_msg)
        
        # Update the indexes to track what we've already published
        self.last_pick_place_index = len(self.sensor_buffer.pick_place_events)
        
        # Publish new pill events
        for i in range(self.last_pill_event_index, len(self.sensor_buffer.pill_events)):
            event = self.sensor_buffer.pill_events[i]
            pill_msg = PillEvent()
            pill_msg.header = header
            
            # Convert ms to ROS time - ensure positive values
            event_secs = max(0, int(event['Time (ms)'] / 1000))
            event_nsecs = max(0, int((event['Time (ms)'] % 1000) * 1e6))
            pill_msg.event_time = rospy.Time(event_secs, event_nsecs)
            
            pill_msg.tilt = event['Tilt']
            pill_msg.roll = event['r']
            pill_msg.pitch = event['p']
            pill_msg.yaw = event['y']
            pill_msg.pressure = event['Pressure']
            pill_msg.light = event['Light']
            
            # Pressure time - ensure positive values
            p_secs = max(0, int(event['Pressure_time'] / 1000))
            p_nsecs = max(0, int((event['Pressure_time'] % 1000) * 1e6))
            pill_msg.pressure_time = rospy.Time(p_secs, p_nsecs)
            
            # Light time - ensure positive values
            l_secs = max(0, int(event['Light_time'] / 1000))
            l_nsecs = max(0, int((event['Light_time'] % 1000) * 1e6))
            pill_msg.light_time = rospy.Time(l_secs, l_nsecs)
            
            # Pick-place times - ensure positive values
            pick_secs = max(0, int(event['pick_time'] / 1000))
            pick_nsecs = max(0, int((event['pick_time'] % 1000) * 1e6))
            place_secs = max(0, int(event['place_time'] / 1000))
            place_nsecs = max(0, int((event['place_time'] % 1000) * 1e6))
            
            pill_msg.pick_time = rospy.Time(pick_secs, pick_nsecs)
            pill_msg.place_time = rospy.Time(place_secs, place_nsecs)
            
            # Check if this pill has been confirmed or forgotten
            pill_msg.confirmed = any(s['pill_event'] is event for s in self.sensor_buffer.swallow_events)
            pill_msg.forgotten = any(f['pill_event'] is event for f in self.sensor_buffer.forget_events)
            
            self.pill_event_pub.publish(pill_msg)
            
        self.last_pill_event_index = len(self.sensor_buffer.pill_events)
        
        # Similarly publish swallow and forget events with the same safety measures
        # Swallow events
        for i in range(self.last_swallow_event_index, len(self.sensor_buffer.swallow_events)):
            event = self.sensor_buffer.swallow_events[i]
            swallow_msg = SwallowEvent()
            swallow_msg.header = header
            
            # Event time - ensure positive values
            event_secs = max(0, int(event['pill_event']['Time (ms)'] / 1000))
            event_nsecs = max(0, int((event['pill_event']['Time (ms)'] % 1000) * 1e6))
            swallow_msg.event_time = rospy.Time(event_secs, event_nsecs)
            
            # Swallow time - ensure positive values
            s_secs = max(0, int(event['swallow_time'] / 1000))
            s_nsecs = max(0, int((event['swallow_time'] % 1000) * 1e6))
            swallow_msg.swallow_time = rospy.Time(s_secs, s_nsecs)
            
            swallow_msg.confirmation_delay = max(0, event['confirmation_delay'] / 1000.0)  # Ensure positive
            
            # Create and attach pill event
            pill_event = event['pill_event']
            pill_msg = self._create_pill_message(pill_event, header)
            swallow_msg.pill_event = pill_msg
            
            self.swallow_event_pub.publish(swallow_msg)
        
        self.last_swallow_event_index = len(self.sensor_buffer.swallow_events)
        
        # Forget events
        for i in range(self.last_forget_event_index, len(self.sensor_buffer.forget_events)):
            event = self.sensor_buffer.forget_events[i]
            forget_msg = ForgetEvent()
            forget_msg.header = header
            
            # Event time - ensure positive values
            event_secs = max(0, int(event['pill_event']['Time (ms)'] / 1000))
            event_nsecs = max(0, int((event['pill_event']['Time (ms)'] % 1000) * 1e6))
            forget_msg.event_time = rospy.Time(event_secs, event_nsecs)
            
            # Forget time - ensure positive values
            f_secs = max(0, int(event['forget_time'] / 1000))
            f_nsecs = max(0, int((event['forget_time'] % 1000) * 1e6))
            forget_msg.forget_time = rospy.Time(f_secs, f_nsecs)
            
            forget_msg.window_duration = max(0, event['window_duration'] / 1000.0)  # Ensure positive
            
            # Create and attach pill event
            pill_event = event['pill_event']
            pill_msg = self._create_pill_message(pill_event, header)
            forget_msg.pill_event = pill_msg
            
            self.forget_event_pub.publish(forget_msg)
        
        self.last_forget_event_index = len(self.sensor_buffer.forget_events)

    def _create_pill_message(self, pill_event, header):
        """Helper method to create a PillEvent message from a pill event dictionary"""
        pill_msg = PillEvent()
        pill_msg.header = header
        
        # Event time - ensure positive values
        event_secs = max(0, int(pill_event['Time (ms)'] / 1000))
        event_nsecs = max(0, int((pill_event['Time (ms)'] % 1000) * 1e6))
        pill_msg.event_time = rospy.Time(event_secs, event_nsecs)
        
        pill_msg.tilt = pill_event['Tilt']
        pill_msg.roll = pill_event['r']
        pill_msg.pitch = pill_event['p']
        pill_msg.yaw = pill_event['y']
        pill_msg.pressure = pill_event['Pressure']
        pill_msg.light = pill_event['Light']
        
        # Pressure time - ensure positive values
        p_secs = max(0, int(pill_event['Pressure_time'] / 1000))
        p_nsecs = max(0, int((pill_event['Pressure_time'] % 1000) * 1e6))
        pill_msg.pressure_time = rospy.Time(p_secs, p_nsecs)
        
        # Light time - ensure positive values
        l_secs = max(0, int(pill_event['Light_time'] / 1000))
        l_nsecs = max(0, int((pill_event['Light_time'] % 1000) * 1e6))
        pill_msg.light_time = rospy.Time(l_secs, l_nsecs)
        
        # Pick-place times - ensure positive values
        pick_secs = max(0, int(pill_event['pick_time'] / 1000))
        pick_nsecs = max(0, int((pill_event['pick_time'] % 1000) * 1e6))
        place_secs = max(0, int(pill_event['place_time'] / 1000))
        place_nsecs = max(0, int((pill_event['place_time'] % 1000) * 1e6))
        
        pill_msg.pick_time = rospy.Time(pick_secs, pick_nsecs)
        pill_msg.place_time = rospy.Time(place_secs, place_nsecs)
        
        # Check if this pill has been confirmed or forgotten
        pill_msg.confirmed = any(s['pill_event'] is pill_event for s in self.sensor_buffer.swallow_events)
        pill_msg.forgotten = any(f['pill_event'] is pill_event for f in self.sensor_buffer.forget_events)
        
        return pill_msg
    
    def _print_summary(self):
        """Print a summary of the simulation results"""
        rospy.loginfo("\n=== Simulation Summary ===")
        rospy.loginfo(f"IMU data points processed: {self.imu_index} / {len(self.imu_data)}")
        rospy.loginfo(f"Pressure/Light data points processed: {self.pl_index} / {len(self.pl_data)}")
        rospy.loginfo(f"Detected {len(self.sensor_buffer.pick_place_events)} pick-place events")
        rospy.loginfo(f"Detected {len(self.sensor_buffer.pill_events)} pill retrieval events")
        
        if self.sensor_buffer.pill_events:
            rospy.loginfo("\nDetailed pill events:")
            for i, event in enumerate(self.sensor_buffer.pill_events):
                time_ms = event['Time (ms)']
                seconds = time_ms / 1000
                minutes = int(seconds // 60)
                seconds = seconds % 60
                
                rospy.loginfo(f"Event {i+1}: Time {minutes}m {seconds:.1f}s")
                rospy.loginfo(f"  - 3D Tilt: {event['Tilt']:.2f}")
                rospy.loginfo(f"  - Orientation: Roll={event['r']:.2f}°, Pitch={event['p']:.2f}°, Yaw={event['y']:.2f}°")
                rospy.loginfo(f"  - Pressure: {event['Pressure']:.0f}")
                rospy.loginfo(f"  - Light: {event['Light']:.2f} lux")
        
        rospy.loginfo(f"\nEvents logged to: {self.log_filename}")
        
    def _force_pill_event(self):
        """Force a pill event for testing purposes"""
        # Only create events at certain intervals
        if not hasattr(self, 'last_forced_event_time') or \
        self.sim_time - self.last_forced_event_time > 10000:  # Every 10 seconds
            
            self.last_forced_event_time = self.sim_time
            
            # Get the latest sensor values
            if (not self.sensor_buffer.r or not self.sensor_buffer.p or 
                not self.sensor_buffer.y or not self.sensor_buffer.pressure or 
                not self.sensor_buffer.light):
                rospy.logwarn("Not enough sensor data to create a forced pill event")
                return
            
            # Make sure pick_time is not negative - use max(0, ...)
            pick_time = max(0, self.sim_time - 2000)  # 2 seconds before or 0 if that would be negative
            
            # Create a forced pill event
            pill_event = {
                'Time (ms)': self.sim_time,
                'Tilt': 10.0,  # Force a high value
                'r': self.sensor_buffer.r[-1],
                'p': self.sensor_buffer.p[-1],
                'y': self.sensor_buffer.y[-1],
                'Pressure': self.sensor_buffer.pressure[-1],
                'Light': self.sensor_buffer.light[-1],
                'Pressure_time': self.sim_time,
                'Light_time': self.sim_time,
                'pick_time': pick_time,  # Using safe value
                'place_time': self.sim_time + 2000,  # 2 seconds after
                'actual_time': datetime.datetime.now(),
                'confirmed': False,
                'forgotten': False
            }
            
            self.sensor_buffer.pill_events.append(pill_event)
            
            # Start swallow confirmation window
            self.sensor_buffer.awaiting_swallow = True
            self.sensor_buffer.pill_awaiting_swallow = pill_event
            self.sensor_buffer.swallow_window_start = pill_event['Time (ms)']
            
            rospy.loginfo(f"\n!!! FORCED PILL EVENT CREATED at {self.sim_time/1000:.1f}s !!!")

if __name__ == '__main__':
    try:
        node = PillDetectionSimulationNode()
        node.start()
    except rospy.ROSInterruptException:
        pass