#!/usr/bin/env python3

import rospy
import numpy as np
import serial
import time
import datetime
import threading
from collections import deque
import logging
import os
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
        self.tilt_threshold = 7
        self.pressure_threshold = 60
        self.light_threshold = 20
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
        """Update derived acceleration metrics based on latest data"""
        if not self.ax:
            return
        
        mag = np.sqrt(self.ax[-1]**2 + self.ay[-1]**2 + self.az[-1]**2)
        self.accel_magnitude.append(mag)
        
        if len(self.accel_magnitude) > 20:
            baseline = np.median(list(self.accel_magnitude)[-20:])
        else:
            baseline = np.median(list(self.accel_magnitude))
        
        self.accel_deviation.append(np.abs(mag - baseline))
        
        if len(self.accel_deviation) >= 5:
            window = list(self.accel_deviation)[-5:]
            smooth_val = np.mean(window)
            self.accel_smooth.append(smooth_val)
        else:
            self.accel_smooth.append(self.accel_deviation[-1])
        
        in_motion = self.accel_smooth[-1] > self.motion_threshold
        self.in_motion.append(in_motion)
        
        if len(self.in_motion) > 1:
            prev_state = 1 if self.in_motion[-2] else 0
            curr_state = 1 if self.in_motion[-1] else 0
            state_change = curr_state - prev_state
        else:
            state_change = 0
        
        self.state_change.append(state_change)
        self._update_pick_place_tracking()
    
    def _update_tilt_metrics(self):
        """Update tilt metrics based on latest orientation data"""
        if not self.r:
            return
        
        r_weight, p_weight, y_weight = 1.0, 1.0, 1.0
        
        r_diff = r_weight * (self.r[-1] - self.rest_r)
        p_diff = p_weight * (self.p[-1] - self.rest_p)
        y_diff = y_weight * (self.y[-1] - self.rest_y)
        
        tilt_mag = np.sqrt(r_diff**2 + p_diff**2 + y_diff**2)
        self.tilt_magnitude.append(tilt_mag)
        
        if len(self.tilt_magnitude) > 1:
            tilt_change = abs(self.tilt_magnitude[-1] - self.tilt_magnitude[-2])
        else:
            tilt_change = 0
        
        self.tilt_change.append(tilt_change)
    
    def _update_pick_place_tracking(self):
        """Update pick-place event tracking based on motion state changes"""
        if not self.state_change:
            return
        
        last_change = self.state_change[-1]
        current_time = self.imu_time[-1]
        
        if last_change == 1:
            self.in_pick_place_event = True
            self.current_pick_time = current_time
            rospy.loginfo(f"Pick up detected at {current_time/1000:.1f}s")
        elif last_change == -1 and self.in_pick_place_event:
            if self.current_pick_time is not None:
                duration = current_time - self.current_pick_time
                
                if duration >= self.min_event_duration:
                    event = {
                        'pick_time': self.current_pick_time,
                        'place_time': current_time,
                        'duration': duration
                    }
                    self.pick_place_events.append(event)
                    rospy.loginfo(f"Place down detected at {current_time/1000:.1f}s (duration: {duration/1000:.1f}s)")
                    
                    self._check_for_pill_event(event)
            
            self.in_pick_place_event = False
            self.current_pick_time = None
    
    def _check_for_pill_event(self, pick_place_event):
        """Check if a pill retrieval event occurred within the pick-place window."""
        pick_time = pick_place_event['pick_time']
        place_time = pick_place_event['place_time']
        
        window_tilt_events = []
        for i in range(len(self.imu_time)):
            time_val = self.imu_time[i]
            if pick_time <= time_val <= place_time:
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
                if i < len(self.pressure) and self.pressure[i] > self.pressure_threshold:
                    window_pressure_events.append({
                        'time': time_val,
                        'pressure': self.pressure[i]
                    })
        
        window_light_events = []
        for i in range(len(self.pl_time)):
            time_val = self.pl_time[i]
            if pick_time <= time_val <= place_time:
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


class PillDetectionNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('pill_detection_node', anonymous=True)
        
        # Get parameters from ROS parameter server
        self.imu_port = rospy.get_param('~imu_port', '/dev/ttyUSB0')  # Default for Linux
        self.imu_baud_rate = rospy.get_param('~imu_baud_rate', 512000)
        self.pl_port = rospy.get_param('~pl_port', '/dev/ttyUSB1')  # Default for Linux
        self.pl_baud_rate = rospy.get_param('~pl_baud_rate', 9600)
        self.log_dir = rospy.get_param('~log_dir', 'pill_logs')
        
        # Set up logging
        self._setup_logging()
        
        # Initialize serial connections
        self.imu_serial, self.pl_serial = self._setup_serial_ports()
        
        # Initialize data buffer
        self.sensor_buffer = SensorDataBuffer(max_size=1000)
        
        # Set up ROS publishers
        self._setup_publishers()
        
        # Initialize sensor reading threads
        self._setup_threads()
        
        # Flag for node status
        self.running = True
        
        rospy.loginfo("Pill detection node initialized")
    
    def _setup_logging(self):
        """Set up logging for pill events"""
        os.makedirs(self.log_dir, exist_ok=True)
        log_filename = os.path.join(
            self.log_dir, 
            f"pill_events_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        logging.basicConfig(
            filename=log_filename, 
            level=logging.INFO,
            format='%(asctime)s - %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.log_filename = log_filename
        rospy.loginfo(f"Logging to {log_filename}")
    
    def _setup_serial_ports(self):
        """Set up serial connections to sensors"""
        imu_serial = None
        pl_serial = None
        
        try:
            imu_serial = serial.Serial(self.imu_port, self.imu_baud_rate, timeout=1)
            rospy.loginfo(f"Connected to IMU sensor on {self.imu_port} with baud rate {self.imu_baud_rate}")
        except Exception as e:
            rospy.logerr(f"Error connecting to IMU sensor: {e}")
        
        try:
            pl_serial = serial.Serial(self.pl_port, self.pl_baud_rate, timeout=1)
            rospy.loginfo(f"Connected to pressure/light sensors on {self.pl_port} with baud rate {self.pl_baud_rate}")
        except Exception as e:
            rospy.logerr(f"Error connecting to pressure/light sensors: {e}")
        
        if imu_serial is None or pl_serial is None:
            rospy.logerr("Failed to connect to one or more sensors. Shutting down.")
            rospy.signal_shutdown("Sensor connection failure")
        
        return imu_serial, pl_serial
    
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
    
    def _setup_threads(self):
        """Set up threads for sensor reading and data publishing"""
        self.imu_thread = threading.Thread(target=self._read_imu_data)
        self.pl_thread = threading.Thread(target=self._read_pl_data)
        self.publish_thread = threading.Thread(target=self._publish_data)
        
        # Set as daemon threads so they exit when the main thread exits
        self.imu_thread.daemon = True
        self.pl_thread.daemon = True
        self.publish_thread.daemon = True
    
    def start(self):
        """Start the node operation"""
        rospy.loginfo("Starting pill detection...")
        
        # Initialize sensors for a few seconds
        self._initialize_sensors()
        
        # Calibrate sensors
        self._calibrate_sensors()
        
        # Start threads
        self.sensor_buffer.start_time = rospy.Time.now().to_nsec() / 1e6  # Convert to milliseconds
        self.imu_thread.start()
        self.pl_thread.start()
        self.publish_thread.start()
        
        rospy.loginfo("Pill detection started")
        rospy.loginfo("\nINSTRUCTIONS:")
        rospy.loginfo("1. When a pill is taken, the system will detect it automatically.")
        rospy.loginfo("2. After taking the pill, PRESS AND HOLD the pressure sensor for 2+ seconds to confirm swallowing.")
        rospy.loginfo("3. If no confirmation within 10 seconds, the pill will be marked as possibly forgotten.")
        
        # Register shutdown hook
        rospy.on_shutdown(self.shutdown)
        
        # Keep the node running
        rospy.spin()
    
    def _initialize_sensors(self):
        """Initialize sensors by discarding initial readings"""
        rospy.loginfo("Initializing sensors...")
        init_start = time.time()
        while time.time() - init_start < 5 and not rospy.is_shutdown():
            if self.imu_serial.in_waiting > 0:
                self.imu_serial.readline()
            if self.pl_serial.in_waiting > 0:
                self.pl_serial.readline()
            time.sleep(0.01)
        rospy.loginfo("Sensor initialization complete")
    
    def _calibrate_sensors(self):
        """Calibrate sensors to detect rest position"""
        rospy.loginfo("Calibrating sensors...")
        calibrated = False
        
        # Read some initial data for calibration
        cal_start = time.time()
        while time.time() - cal_start < 5 and not rospy.is_shutdown() and not calibrated:
            if self.imu_serial.in_waiting > 0:
                line = self.imu_serial.readline()
                imu_data = self._parse_imu_data(line)
                
                if imu_data:
                    ax_val, ay_val, az_val, r_val, p_val, y_val = imu_data
                    self.sensor_buffer.add_imu_reading(ax_val, ay_val, az_val, r_val, p_val, y_val)
                    calibrated = self.sensor_buffer.calibrate_rest_position()
            
            time.sleep(0.01)
        
        if calibrated:
            rospy.loginfo("Sensor calibration complete")
        else:
            rospy.logwarn("Sensor calibration timeout - using default values")
            self.sensor_buffer.rest_r = 0
            self.sensor_buffer.rest_p = 0
            self.sensor_buffer.rest_y = 80
    
    def _read_imu_data(self):
        """Thread function to read IMU data"""
        while self.running and not rospy.is_shutdown():
            try:
                if self.imu_serial.in_waiting > 0:
                    line = self.imu_serial.readline()
                    imu_data = self._parse_imu_data(line)
                    
                    if imu_data:
                        ax_val, ay_val, az_val, r_val, p_val, y_val = imu_data
                        self.sensor_buffer.add_imu_reading(ax_val, ay_val, az_val, r_val, p_val, y_val)
                
                time.sleep(0.001)  # Small sleep to prevent CPU hogging
            except Exception as e:
                rospy.logerr(f"Error reading IMU data: {e}")
                time.sleep(1)
    
    def _read_pl_data(self):
        """Thread function to read pressure/light data"""
        while self.running and not rospy.is_shutdown():
            try:
                if self.pl_serial.in_waiting > 0:
                    line = self.pl_serial.readline()
                    pl_data = self._parse_pl_data(line)
                    
                    if pl_data:
                        pressure_val, light_val = pl_data
                        self.sensor_buffer.add_pl_reading(pressure_val, light_val)
                
                time.sleep(0.001)  # Small sleep to prevent CPU hogging
            except Exception as e:
                rospy.logerr(f"Error reading pressure/light data: {e}")
                time.sleep(1)
    
    def _parse_imu_data(self, line):
        """Parse IMU data from serial line"""
        try:
            if isinstance(line, bytes):
                line = line.decode('utf-8').strip()
            else:
                line = line.strip()
            if not line:
                return None
            values = [float(x) for x in line.split(',')]
            if len(values) == 6:
                return tuple(values)
            else:
                rospy.logwarn(f"Incomplete IMU data: {line}")
                return None
        except Exception as e:
            rospy.logwarn(f"Error parsing IMU data: {e} | Line: {line}")
            return None
    
    def _parse_pl_data(self, line):
        """Parse pressure and light data from serial line"""
        try:
            if isinstance(line, bytes):
                line = line.decode('utf-8').strip()
            else:
                line = line.strip()
            if not line:
                return None
            parts = line.split()
            if len(parts) >= 3:
                pressure = float(parts[1])
                light = float(parts[2])
                return pressure, light
            else:
                rospy.logwarn(f"Incomplete pressure/light data: {line}")
                return None
        except Exception as e:
            rospy.logwarn(f"Error parsing pressure/light data: {e} | Line: {line}")
            return None
    
    def _publish_data(self):
        """Thread function to publish sensor data and events to ROS topics"""
        last_publish_time = 0
        publish_interval = 0.05  # 20 Hz
        
        # Track the last published event indices to avoid republishing
        last_pick_place_index = 0
        last_pill_event_index = 0
        last_swallow_event_index = 0
        last_forget_event_index = 0
        
        while self.running and not rospy.is_shutdown():
            current_time = time.time()
            
            if current_time - last_publish_time >= publish_interval:
                self._publish_sensor_data()
                
                # Publish new events
                self._publish_new_events(
                    last_pick_place_index,
                    last_pill_event_index,
                    last_swallow_event_index,
                    last_forget_event_index
                )
                
                # Update indices
                last_pick_place_index = len(self.sensor_buffer.pick_place_events)
                last_pill_event_index = len(self.sensor_buffer.pill_events)
                last_swallow_event_index = len(self.sensor_buffer.swallow_events)
                last_forget_event_index = len(self.sensor_buffer.forget_events)
                
                last_publish_time = current_time
            
            time.sleep(0.01)
    
    def _publish_sensor_data(self):
        """Publish latest sensor data to ROS topics"""
        if not self.sensor_buffer.imu_time:
            return
        
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
            # For accurate conversion, use tf.transformations.quaternion_from_euler
            # But we're keeping dependencies minimal here
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
    
    def _publish_new_events(self, last_pick_place_idx, last_pill_idx, last_swallow_idx, last_forget_idx):
        """Publish any new events that have occurred"""
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "pill_sensor"
        
        # Publish new pick-place events
        for i in range(last_pick_place_idx, len(self.sensor_buffer.pick_place_events)):
            event = self.sensor_buffer.pick_place_events[i]
            pp_msg = PickPlaceEvent()
            pp_msg.header = header
            
            # Convert ms to ROS time
            pick_secs = int(event['pick_time'] / 1000)
            pick_nsecs = int((event['pick_time'] % 1000) * 1e6)
            place_secs = int(event['place_time'] / 1000)
            place_nsecs = int((event['place_time'] % 1000) * 1e6)
            
            pp_msg.pick_time = rospy.Time(pick_secs, pick_nsecs)
            pp_msg.place_time = rospy.Time(place_secs, place_nsecs)
            pp_msg.duration = event['duration'] / 1000.0  # Convert to seconds
            
            self.pick_place_pub.publish(pp_msg)
        
        # Publish new pill events
        for i in range(last_pill_idx, len(self.sensor_buffer.pill_events)):
            event = self.sensor_buffer.pill_events[i]
            pill_msg = PillEvent()
            pill_msg.header = header
            
            # Convert ms to ROS time
            event_secs = int(event['Time (ms)'] / 1000)
            event_nsecs = int((event['Time (ms)'] % 1000) * 1e6)
            pill_msg.event_time = rospy.Time(event_secs, event_nsecs)
            
            pill_msg.tilt = event['Tilt']
            pill_msg.roll = event['r']
            pill_msg.pitch = event['p']
            pill_msg.yaw = event['y']
            pill_msg.pressure = event['Pressure']
            pill_msg.light = event['Light']
            
            # Pressure time
            p_secs = int(event['Pressure_time'] / 1000)
            p_nsecs = int((event['Pressure_time'] % 1000) * 1e6)
            pill_msg.pressure_time = rospy.Time(p_secs, p_nsecs)
            
            # Light time
            l_secs = int(event['Light_time'] / 1000)
            l_nsecs = int((event['Light_time'] % 1000) * 1e6)
            pill_msg.light_time = rospy.Time(l_secs, l_nsecs)
            
            # Pick-place times
            pick_secs = int(event['pick_time'] / 1000)
            pick_nsecs = int((event['pick_time'] % 1000) * 1e6)
            place_secs = int(event['place_time'] / 1000)
            place_nsecs = int((event['place_time'] % 1000) * 1e6)
            
            pill_msg.pick_time = rospy.Time(pick_secs, pick_nsecs)
            pill_msg.place_time = rospy.Time(place_secs, place_nsecs)
            
            # Check if this pill has been confirmed or forgotten
            pill_msg.confirmed = any(s['pill_event'] is event for s in self.sensor_buffer.swallow_events)
            pill_msg.forgotten = any(f['pill_event'] is event for f in self.sensor_buffer.forget_events)
            
            self.pill_event_pub.publish(pill_msg)
        
        # Publish new swallow events
        for i in range(last_swallow_idx, len(self.sensor_buffer.swallow_events)):
            event = self.sensor_buffer.swallow_events[i]
            swallow_msg = SwallowEvent()
            swallow_msg.header = header
            
            # Event time (from the pill event)
            event_secs = int(event['pill_event']['Time (ms)'] / 1000)
            event_nsecs = int((event['pill_event']['Time (ms)'] % 1000) * 1e6)
            swallow_msg.event_time = rospy.Time(event_secs, event_nsecs)
            
            # Swallow time
            s_secs = int(event['swallow_time'] / 1000)
            s_nsecs = int((event['swallow_time'] % 1000) * 1e6)
            swallow_msg.swallow_time = rospy.Time(s_secs, s_nsecs)
            
            swallow_msg.confirmation_delay = event['confirmation_delay'] / 1000.0  # Convert to seconds
            
            # Create and attach the pill event
            pill_event = event['pill_event']
            pill_msg = PillEvent()
            pill_msg.header = header
            
            # Convert ms to ROS time
            pill_secs = int(pill_event['Time (ms)'] / 1000)
            pill_nsecs = int((pill_event['Time (ms)'] % 1000) * 1e6)
            pill_msg.event_time = rospy.Time(pill_secs, pill_nsecs)
            
            pill_msg.tilt = pill_event['Tilt']
            pill_msg.roll = pill_event['r']
            pill_msg.pitch = pill_event['p']
            pill_msg.yaw = pill_event['y']
            pill_msg.pressure = pill_event['Pressure']
            pill_msg.light = pill_event['Light']
            
            # Pressure time
            p_secs = int(pill_event['Pressure_time'] / 1000)
            p_nsecs = int((pill_event['Pressure_time'] % 1000) * 1e6)
            pill_msg.pressure_time = rospy.Time(p_secs, p_nsecs)
            
            # Light time
            l_secs = int(pill_event['Light_time'] / 1000)
            l_nsecs = int((pill_event['Light_time'] % 1000) * 1e6)
            pill_msg.light_time = rospy.Time(l_secs, l_nsecs)
            
            # Pick-place times
            pick_secs = int(pill_event['pick_time'] / 1000)
            pick_nsecs = int((pill_event['pick_time'] % 1000) * 1e6)
            place_secs = int(pill_event['place_time'] / 1000)
            place_nsecs = int((pill_event['place_time'] % 1000) * 1e6)
            
            pill_msg.pick_time = rospy.Time(pick_secs, pick_nsecs)
            pill_msg.place_time = rospy.Time(place_secs, place_nsecs)
            
            pill_msg.confirmed = True
            pill_msg.forgotten = False
            
            swallow_msg.pill_event = pill_msg
            
            self.swallow_event_pub.publish(swallow_msg)
        
        # Publish new forget events
        for i in range(last_forget_idx, len(self.sensor_buffer.forget_events)):
            event = self.sensor_buffer.forget_events[i]
            forget_msg = ForgetEvent()
            forget_msg.header = header
            
            # Event time (from the pill event)
            event_secs = int(event['pill_event']['Time (ms)'] / 1000)
            event_nsecs = int((event['pill_event']['Time (ms)'] % 1000) * 1e6)
            forget_msg.event_time = rospy.Time(event_secs, event_nsecs)
            
            # Forget time
            f_secs = int(event['forget_time'] / 1000)
            f_nsecs = int((event['forget_time'] % 1000) * 1e6)
            forget_msg.forget_time = rospy.Time(f_secs, f_nsecs)
            
            forget_msg.window_duration = event['window_duration'] / 1000.0  # Convert to seconds
            
            # Create and attach the pill event
            pill_event = event['pill_event']
            pill_msg = PillEvent()
            pill_msg.header = header
            
            # Convert ms to ROS time
            pill_secs = int(pill_event['Time (ms)'] / 1000)
            pill_nsecs = int((pill_event['Time (ms)'] % 1000) * 1e6)
            pill_msg.event_time = rospy.Time(pill_secs, pill_nsecs)
            
            pill_msg.tilt = pill_event['Tilt']
            pill_msg.roll = pill_event['r']
            pill_msg.pitch = pill_event['p']
            pill_msg.yaw = pill_event['y']
            pill_msg.pressure = pill_event['Pressure']
            pill_msg.light = pill_event['Light']
            
            # Pressure time
            p_secs = int(pill_event['Pressure_time'] / 1000)
            p_nsecs = int((pill_event['Pressure_time'] % 1000) * 1e6)
            pill_msg.pressure_time = rospy.Time(p_secs, p_nsecs)
            
            # Light time
            l_secs = int(pill_event['Light_time'] / 1000)
            l_nsecs = int((pill_event['Light_time'] % 1000) * 1e6)
            pill_msg.light_time = rospy.Time(l_secs, l_nsecs)
            
            # Pick-place times
            pick_secs = int(pill_event['pick_time'] / 1000)
            pick_nsecs = int((pill_event['pick_time'] % 1000) * 1e6)
            place_secs = int(pill_event['place_time'] / 1000)
            place_nsecs = int((pill_event['place_time'] % 1000) * 1e6)
            
            pill_msg.pick_time = rospy.Time(pick_secs, pick_nsecs)
            pill_msg.place_time = rospy.Time(place_secs, place_nsecs)
            
            pill_msg.confirmed = False
            pill_msg.forgotten = True
            
            forget_msg.pill_event = pill_msg
            
            self.forget_event_pub.publish(forget_msg)
    
    def shutdown(self):
        """Clean shutdown of the node"""
        rospy.loginfo("Shutting down pill detection node...")
        self.running = False
        
        # Close serial connections
        if hasattr(self, 'imu_serial') and self.imu_serial:
            self.imu_serial.close()
        if hasattr(self, 'pl_serial') and self.pl_serial:
            self.pl_serial.close()
        
        # Print summary
        rospy.loginfo("\n=== Monitoring Summary ===")
        rospy.loginfo(f"Detected {len(self.sensor_buffer.pick_place_events)} pick-place events")
        rospy.loginfo(f"Detected {len(self.sensor_buffer.pill_events)} pill retrieval events")
        rospy.loginfo(f"Confirmed {len(self.sensor_buffer.swallow_events)} pill swallow events")
        rospy.loginfo(f"Possibly forgotten pills: {len(self.sensor_buffer.forget_events)}")
        
        if self.sensor_buffer.pill_events:
            rospy.loginfo("\nDetailed pill events:")
            for i, event in enumerate(self.sensor_buffer.pill_events):
                time_str = event['actual_time'].strftime('%Y-%m-%d %H:%M:%S')
                
                # Check if this pill was confirmed swallowed
                confirmed = any(s['pill_event'] is event for s in self.sensor_buffer.swallow_events)
                forgotten = any(f['pill_event'] is event for f in self.sensor_buffer.forget_events)
                
                status = "CONFIRMED SWALLOWED" if confirmed else "POSSIBLY FORGOTTEN" if forgotten else "UNCONFIRMED"
                rospy.loginfo(f"Event {i+1}: {time_str} - Status: {status}")
        
        rospy.loginfo(f"\nEvents logged to: {self.log_filename}")


if __name__ == '__main__':
    try:
        node = PillDetectionNode()
        node.start()
    except rospy.ROSInterruptException:
        pass