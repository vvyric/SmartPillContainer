import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks
import serial
import time
import datetime
import threading
from collections import deque
import logging
import os

# Set up logging
log_dir = "pill_logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"pill_events_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(filename=log_filename, level=logging.INFO,
                   format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Configure serial ports with updated settings:
# IMU sensor: COM8, baud rate 512000; Pressure/Light sensor: COM6, baud rate 9600.
def setup_serial_ports(imu_port='COM8', imu_baud_rate=512000, pl_port='COM6', pl_baud_rate=9600):
    """Set up serial connections to the IMU and pressure/light sensors."""
    try:
        imu_serial = serial.Serial(imu_port, imu_baud_rate, timeout=1)
        print(f"Connected to IMU sensor on {imu_port} with baud rate {imu_baud_rate}")
    except Exception as e:
        print(f"Error connecting to IMU sensor: {e}")
        imu_serial = None
    
    try:
        pl_serial = serial.Serial(pl_port, pl_baud_rate, timeout=1)
        print(f"Connected to pressure/light sensors on {pl_port} with baud rate {pl_baud_rate}")
    except Exception as e:
        print(f"Error connecting to pressure/light sensors: {e}")
        pl_serial = None
    
    return imu_serial, pl_serial

# Data buffers for real-time processing
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
        
    def add_pl_reading(self, pressure, light):
        """Add a new pressure/light sensor reading to the buffer"""
        current_time = self.get_relative_time()
        
        self.pl_time.append(current_time)
        self.pressure.append(pressure)
        self.light.append(light)
    
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
            print(f"Pick up detected at {current_time/1000:.1f}s")
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
                    print(f"Place down detected at {current_time/1000:.1f}s (duration: {duration/1000:.1f}s)")
                    
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
                'actual_time': datetime.datetime.now()
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
                f"  - Light: {pill_event['Light']:.2f} lux at {pill_event['Light_time']/1000:.1f}s"
            )
            
            print(f"\n!!! PILL EVENT DETECTED !!!\n{log_message}")
            logging.info(f"PILL EVENT DETECTED\n{log_message}")
            logging.info(f"Actual time of event: {pill_event['actual_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    def calibrate_rest_position(self, num_samples=20):
        """Calibrate the rest position of the device using initial readings"""
        if len(self.r) >= num_samples and len(self.p) >= num_samples and len(self.y) >= num_samples:
            self.rest_r = np.median(list(self.r)[:num_samples])
            self.rest_p = np.median(list(self.p)[:num_samples])
            self.rest_y = np.median(list(self.y)[:num_samples])
            
            print(f"Calibrated rest position: r={self.rest_r:.2f}, p={self.rest_p:.2f}, y={self.rest_y:.2f}")
            return True
        return False

# Serial data parsing functions
def parse_imu_data(line):
    """
    Parse IMU data from serial line.
    Expected format: "ax,ay,az,r,p,y"
    """
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
            print(f"Warning: Incomplete IMU data: {line}")
            return None
    except Exception as e:
        print(f"Error parsing IMU data: {e} | Line: {line}")
        return None

def parse_pl_data(line):
    """
    Parse pressure and light data from serial line.
    Expected format: "time(ms) pressure light unit"
    """
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
            print(f"Warning: Incomplete pressure/light data: {line}")
            return None
    except Exception as e:
        print(f"Error parsing pressure/light data: {e} | Line: {line}")
        return None

# Live visualization setup
def setup_live_visualization():
    """Setup figures for real-time visualization"""
    plt.ion()
    fig = plt.figure(figsize=(12, 8))
    
    ax1 = fig.add_subplot(411)  # Acceleration
    ax2 = fig.add_subplot(412)  # Orientation
    ax3 = fig.add_subplot(413)  # Pressure
    ax4 = fig.add_subplot(414)  # Light
    
    ax1.set_title('Acceleration')
    ax1.set_ylabel('Magnitude')
    ax1.grid(True)
    
    ax2.set_title('Orientation')
    ax2.set_ylabel('Degrees')
    ax2.grid(True)
    
    ax3.set_title('Pressure')
    ax3.set_ylabel('Value')
    ax3.grid(True)
    
    ax4.set_title('Light')
    ax4.set_ylabel('Lux')
    ax4.set_xlabel('Time (s)')
    ax4.grid(True)
    
    plt.tight_layout()
    return fig, (ax1, ax2, ax3, ax4)

def update_visualization(fig, axes, sensor_buffer, max_time_window=60):
    """Update the live visualization with latest data"""
    ax1, ax2, ax3, ax4 = axes
    
    imu_time = np.array(list(sensor_buffer.imu_time)) / 1000
    accel_mag = np.array(list(sensor_buffer.accel_magnitude))
    accel_smooth = np.array(list(sensor_buffer.accel_smooth))
    roll = np.array(list(sensor_buffer.r))
    pitch = np.array(list(sensor_buffer.p))
    yaw = np.array(list(sensor_buffer.y))
    tilt_mag = np.array(list(sensor_buffer.tilt_magnitude))
    
    pl_time = np.array(list(sensor_buffer.pl_time)) / 1000
    pressure = np.array(list(sensor_buffer.pressure))
    light = np.array(list(sensor_buffer.light))
    
    for ax in axes:
        ax.clear()
    
    if len(imu_time) > 0:
        current_time = imu_time[-1]
        x_min = max(0, current_time - max_time_window)
        for ax in axes:
            ax.set_xlim(x_min, current_time + 2)
    
    if len(imu_time) > 0:
        ax1.plot(imu_time, accel_mag, 'k-', label="Accel Magnitude", alpha=0.5)
        ax1.plot(imu_time, accel_smooth, 'r-', label="Smoothed", alpha=0.7)
        ax1.axhline(y=sensor_buffer.motion_threshold, color='g', linestyle='--', alpha=0.7, label="Threshold")
        ax1.set_title('Acceleration')
        ax1.set_ylabel('Magnitude')
        ax1.grid(True)
        ax1.legend(loc='upper left')
    
    if len(imu_time) > 0:
        ax2.plot(imu_time, roll, 'r-', label="Roll", alpha=0.7)
        ax2.plot(imu_time, pitch, 'g-', label="Pitch", alpha=0.7)
        ax2.plot(imu_time, yaw, 'b-', label="Yaw", alpha=0.7)
        ax2.plot(imu_time, tilt_mag, 'k-', label="Tilt Mag", alpha=0.5)
        ax2.axhline(y=sensor_buffer.tilt_threshold, color='r', linestyle='--', alpha=0.7, label="Threshold")
        ax2.set_title('Orientation')
        ax2.set_ylabel('Degrees')
        ax2.grid(True)
        ax2.legend(loc='upper left')
    
    if len(pl_time) > 0:
        ax3.plot(pl_time, pressure, 'b-')
        ax3.axhline(y=sensor_buffer.pressure_threshold, color='r', linestyle='--', alpha=0.7, label="Threshold")
        ax3.set_title('Pressure')
        ax3.set_ylabel('Value')
        ax3.grid(True)
        ax3.legend(loc='upper left')
    
    if len(pl_time) > 0:
        ax4.plot(pl_time, light, 'r-')
        ax4.axhline(y=sensor_buffer.light_threshold, color='g', linestyle='--', alpha=0.7, label="Threshold")
        ax4.set_title('Light')
        ax4.set_ylabel('Lux')
        ax4.set_xlabel('Time (s)')
        ax4.grid(True)
        ax4.legend(loc='upper left')
    
    for pp_event in sensor_buffer.pick_place_events:
        pick_time = pp_event['pick_time'] / 1000
        place_time = pp_event['place_time'] / 1000
        
        has_pill_event = any(
            pick_time <= pe['Time (ms)'] / 1000 <= place_time 
            for pe in sensor_buffer.pill_events
        )
        
        color = 'green' if has_pill_event else 'yellow'
        alpha = 0.2
        
        for ax in axes:
            y_min, y_max = ax.get_ylim()
            height = y_max - y_min
            rect = Rectangle((pick_time, y_min), width=(place_time - pick_time), height=height, color=color, alpha=alpha)
            ax.add_patch(rect)
    
    for event in sensor_buffer.pill_events:
        event_time = event['Time (ms)'] / 1000
        for ax in axes:
            ax.axvline(x=event_time, color='m', linestyle='-', alpha=0.9, linewidth=2)
    
    fig.canvas.draw_idle()
    plt.pause(0.1)

# Main real-time monitoring function with separate baud rates for IMU and pressure/light sensors
def monitor_sensors(imu_port='COM8', imu_baud_rate=512000, pl_port='COM6', pl_baud_rate=9600, visualize=True):
    """
    Monitor sensors in real-time and detect pill retrieval events.
    
    Args:
        imu_port: Serial port for IMU sensor (default COM8)
        imu_baud_rate: Baud rate for IMU sensor (default 512000)
        pl_port: Serial port for pressure/light sensor (default COM6)
        pl_baud_rate: Baud rate for pressure/light sensor (default 9600)
        visualize: Whether to show real-time visualization
    """
    imu_serial, pl_serial = setup_serial_ports(imu_port, imu_baud_rate, pl_port, pl_baud_rate)
    
    if imu_serial is None or pl_serial is None:
        print("Error connecting to one or more sensors. Aborting.")
        return
    
    sensor_buffer = SensorDataBuffer(max_size=1000)
    
    fig = None
    axes = None
    if visualize:
        fig, axes = setup_live_visualization()
    
    last_viz_update = time.time()
    viz_update_interval = 0.5
    
    calibrated = False
    
    try:
        print("Starting real-time monitoring...")
        print("Press Ctrl+C to stop")
        
        print("Initializing sensors...")
        init_start = time.time()
        while time.time() - init_start < 5:
            if imu_serial.in_waiting > 0:
                imu_serial.readline()
            if pl_serial.in_waiting > 0:
                pl_serial.readline()
        
        print("Initialization complete. Starting detection...")
        sensor_buffer.start_time = time.time() * 1000
        
        while True:
            if imu_serial.in_waiting > 0:
                line = imu_serial.readline()
                imu_data = parse_imu_data(line)
                
                if imu_data:
                    ax_val, ay_val, az_val, r_val, p_val, y_val = imu_data
                    sensor_buffer.add_imu_reading(ax_val, ay_val, az_val, r_val, p_val, y_val)
                    
                    if not calibrated:
                        calibrated = sensor_buffer.calibrate_rest_position()
                        if calibrated:
                            print("Sensor calibration complete.")
            
            if pl_serial.in_waiting > 0:
                line = pl_serial.readline()
                pl_data = parse_pl_data(line)
                
                if pl_data:
                    pressure_val, light_val = pl_data
                    sensor_buffer.add_pl_reading(pressure_val, light_val)
            
            if visualize and time.time() - last_viz_update > viz_update_interval:
                update_visualization(fig, axes, sensor_buffer)
                last_viz_update = time.time()
            
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    
    finally:
        if imu_serial:
            imu_serial.close()
        if pl_serial:
            pl_serial.close()
        
        print("\n=== Monitoring Summary ===")
        print(f"Detected {len(sensor_buffer.pick_place_events)} pick-place events")
        print(f"Detected {len(sensor_buffer.pill_events)} pill retrieval events")
        
        if sensor_buffer.pill_events:
            print("\nDetailed pill events:")
            for i, event in enumerate(sensor_buffer.pill_events):
                time_str = event['actual_time'].strftime('%Y-%m-%d %H:%M:%S')
                print(f"Event {i+1}: {time_str}")
        
        print(f"\nEvents logged to: {log_filename}")
        
        if visualize:
            print("Close the visualization window to exit")
            plt.ioff()
            plt.show()

# If run as script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time pill retrieval event detection")
    parser.add_argument("--imu-port", default="COM8", help="Serial port for IMU sensor")
    parser.add_argument("--imu-baud", type=int, default=512000, help="Baud rate for IMU sensor")
    parser.add_argument("--pl-port", default="COM6", help="Serial port for pressure/light sensor")
    parser.add_argument("--pl-baud", type=int, default=9600, help="Baud rate for pressure/light sensor")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    
    args = parser.parse_args()
    
    monitor_sensors(
        imu_port=args.imu_port,
        imu_baud_rate=args.imu_baud,
        pl_port=args.pl_port,
        pl_baud_rate=args.pl_baud,
        visualize=not args.no_viz
    )
