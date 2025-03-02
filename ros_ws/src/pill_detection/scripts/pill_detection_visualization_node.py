#!/usr/bin/env python3

import rospy
import numpy as np
# Set matplotlib backend explicitly before other imports
import matplotlib
matplotlib.use('TkAgg')  # Try different backends if this doesn't work: 'Qt5Agg', 'Agg', etc.
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
from collections import deque
import threading
from std_msgs.msg import Header
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
from pill_detection.msg import PressureLight, PillEvent, SwallowEvent, ForgetEvent, PickPlaceEvent

class PillDetectionVisualizer:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('pill_detection_visualizer', anonymous=True)
        
        # Set start time - initialize this BEFORE setting up subscribers
        self.start_time = rospy.Time.now().to_sec()
        
        # Get parameters
        self.max_time_window = rospy.get_param('~max_time_window', 60)  # Time window in seconds
        self.buffer_size = rospy.get_param('~buffer_size', 5000)  # Max data points to keep
        self.update_interval = rospy.get_param('~update_interval', 500)  # Milliseconds
        
        # Initialize data buffers with deque for efficient appending and popping
        self._init_data_buffers()
        
        # Subscribe to topics
        self._init_subscribers()
        
        # Initialize figures and axes in a separate thread
        self.plot_initialized = False
        threading.Thread(target=self._init_plot, daemon=True).start()
        
        rospy.loginfo("Visualization node started")
    
    def _init_plot(self):
        """Initialize plot in a separate thread"""
        try:
            # Initialize figures and axes
            self.fig, self.axes = self._setup_visualization()
            
            # Animation object
            self.ani = animation.FuncAnimation(
                self.fig, self._update_plots, interval=self.update_interval, 
                blit=False, cache_frame_data=False
            )
            
            self.plot_initialized = True
            rospy.loginfo("Plot initialization complete")
            
            # This line blocks until the window is closed
            plt.show()
        except Exception as e:
            rospy.logerr(f"Error initializing plot: {e}")
            # Try a different backend if this one fails
            try:
                matplotlib.use('Agg')
                rospy.logwarn("Trying fallback matplotlib backend 'Agg' (non-interactive)")
                self.fig, self.axes = self._setup_visualization()
                # Save plots instead of showing them
                self._periodic_save_plots()
            except Exception as e2:
                rospy.logerr(f"Error with fallback plotting: {e2}")
    
    def _periodic_save_plots(self):
        """Periodically save plots instead of interactive display"""
        if hasattr(self, 'fig') and self.fig is not None:
            try:
                # Update the plots
                self._update_plots(None)
                # Save the figure
                timestamp = rospy.Time.now().to_sec()
                filename = f"/tmp/pill_plot_{timestamp:.0f}.png"
                self.fig.savefig(filename)
                rospy.loginfo(f"Saved plot to {filename}")
            except Exception as e:
                rospy.logwarn(f"Error saving plot: {e}")
        
        # Schedule the next update
        threading.Timer(5.0, self._periodic_save_plots).start()

    
    def _init_data_buffers(self):
        """Initialize data buffers for all visualization data"""
        # IMU data
        self.imu_time = deque(maxlen=self.buffer_size)
        self.ax = deque(maxlen=self.buffer_size)
        self.ay = deque(maxlen=self.buffer_size)
        self.az = deque(maxlen=self.buffer_size)
        
        # Orientation data
        self.roll = deque(maxlen=self.buffer_size)
        self.pitch = deque(maxlen=self.buffer_size)
        self.yaw = deque(maxlen=self.buffer_size)
        
        # Derived data
        self.accel_mag = deque(maxlen=self.buffer_size)
        self.accel_smooth = deque(maxlen=self.buffer_size)
        self.in_motion = deque(maxlen=self.buffer_size)
        self.tilt_mag = deque(maxlen=self.buffer_size)
        self.tilt_change = deque(maxlen=self.buffer_size)
        
        # Pressure/Light data
        self.pl_time = deque(maxlen=self.buffer_size)
        self.pressure = deque(maxlen=self.buffer_size)
        self.light = deque(maxlen=self.buffer_size)
        
        # Event data
        self.pick_place_events = []
        self.pill_events = []
        self.swallow_events = []
        self.forget_events = []
        
        # Thresholds (will be set based on observed values)
        self.motion_threshold = 0.02
        self.tilt_threshold = 7
        self.pressure_threshold = 60
        self.light_threshold = 20
    
    def _init_subscribers(self):
        """Initialize all the ROS subscribers"""
        # Subscribe to raw sensor data
        rospy.Subscriber("pill_detection/imu", Imu, self._imu_callback)
        rospy.Subscriber("pill_detection/pressure_light", PressureLight, self._pl_callback)
        
        # Subscribe to derived metrics
        rospy.Subscriber("pill_detection/accel_magnitude", Vector3, self._accel_mag_callback)
        rospy.Subscriber("pill_detection/tilt_magnitude", Vector3, self._tilt_mag_callback)
        
        # Subscribe to events
        rospy.Subscriber("pill_detection/pick_place_events", PickPlaceEvent, self._pick_place_callback)
        rospy.Subscriber("pill_detection/pill_events", PillEvent, self._pill_event_callback)
        rospy.Subscriber("pill_detection/swallow_events", SwallowEvent, self._swallow_event_callback)
        rospy.Subscriber("pill_detection/forget_events", ForgetEvent, self._forget_event_callback)
    
    def _imu_callback(self, msg):
        """Callback for IMU data"""
        current_time = self._get_relative_time(msg.header.stamp)
        
        # Extract linear acceleration
        self.imu_time.append(current_time)
        self.ax.append(msg.linear_acceleration.x)
        self.ay.append(msg.linear_acceleration.y)
        self.az.append(msg.linear_acceleration.z)
        
        # Extract orientation (roll, pitch, yaw from quaternion)
        # This is a simplified conversion from quaternion to Euler angles
        # For more accurate conversion, consider using tf transformations
        q = msg.orientation
        
        # Calculate roll (x-axis rotation)
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Calculate pitch (y-axis rotation)
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1:
            pitch = np.pi / 2.0 if sinp > 0 else -np.pi / 2.0
        else:
            pitch = np.arcsin(sinp)
        
        # Calculate yaw (z-axis rotation)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        # Convert to degrees
        self.roll.append(np.degrees(roll))
        self.pitch.append(np.degrees(pitch))
        self.yaw.append(np.degrees(yaw))
    
    def _pl_callback(self, msg):
        """Callback for pressure and light data"""
        current_time = self._get_relative_time(msg.header.stamp)
        
        self.pl_time.append(current_time)
        self.pressure.append(msg.pressure)
        self.light.append(msg.light)
    
    def _accel_mag_callback(self, msg):
        """Callback for acceleration magnitude data"""
        # X component is the raw acceleration magnitude
        self.accel_mag.append(msg.x)
        
        # Y component is the smoothed acceleration
        self.accel_smooth.append(msg.y)
        
        # Z component is a boolean flag for motion
        self.in_motion.append(msg.z > 0.5)
        
        # Update the motion threshold if we receive non-zero values
        if msg.y > 0:
            # Dynamic adjustment of threshold based on observed values
            self.motion_threshold = max(0.02, msg.y * 0.5)
    
    def _tilt_mag_callback(self, msg):
        """Callback for tilt magnitude data"""
        # X component is the tilt magnitude
        self.tilt_mag.append(msg.x)
        
        # Y component is the tilt change (derivative)
        self.tilt_change.append(msg.y)
        
        # Update the tilt threshold if we receive non-zero values
        if msg.y > 0:
            # Dynamic adjustment of threshold based on observed values
            self.tilt_threshold = max(7, msg.y * 0.5)
    
    def _pick_place_callback(self, msg):
        """Callback for pick-place events"""
        pick_time = self._get_relative_time(msg.pick_time)
        place_time = self._get_relative_time(msg.place_time)
        
        event = {
            'pick_time': pick_time,
            'place_time': place_time,
            'duration': msg.duration
        }
        
        self.pick_place_events.append(event)
        rospy.loginfo(f"Pick-Place event: {pick_time:.1f}s to {place_time:.1f}s (duration: {msg.duration:.1f}s)")
    
    def _pill_event_callback(self, msg):
        """Callback for pill events"""
        event_time = self._get_relative_time(msg.event_time)
        
        event = {
            'time': event_time,
            'tilt': msg.tilt,
            'roll': msg.roll,
            'pitch': msg.pitch,
            'yaw': msg.yaw,
            'pressure': msg.pressure,
            'light': msg.light,
            'pressure_time': self._get_relative_time(msg.pressure_time),
            'light_time': self._get_relative_time(msg.light_time),
            'pick_time': self._get_relative_time(msg.pick_time),
            'place_time': self._get_relative_time(msg.place_time),
            'confirmed': msg.confirmed,
            'forgotten': msg.forgotten
        }
        
        self.pill_events.append(event)
        rospy.loginfo(f"Pill event detected at {event_time:.1f}s")
    
    def _swallow_event_callback(self, msg):
        """Callback for swallow confirmation events"""
        event_time = self._get_relative_time(msg.event_time)
        swallow_time = self._get_relative_time(msg.swallow_time)
        
        event = {
            'event_time': event_time,
            'swallow_time': swallow_time,
            'confirmation_delay': msg.confirmation_delay,
            'pill_event': self._convert_pill_event_msg(msg.pill_event)
        }
        
        self.swallow_events.append(event)
        rospy.loginfo(f"Swallow confirmed at {swallow_time:.1f}s (delay: {msg.confirmation_delay:.1f}s)")
    
    def _forget_event_callback(self, msg):
        """Callback for forgotten pill events"""
        event_time = self._get_relative_time(msg.event_time)
        forget_time = self._get_relative_time(msg.forget_time)
        
        event = {
            'event_time': event_time,
            'forget_time': forget_time,
            'window_duration': msg.window_duration,
            'pill_event': self._convert_pill_event_msg(msg.pill_event)
        }
        
        self.forget_events.append(event)
        rospy.loginfo(f"Possible forgotten pill at {forget_time:.1f}s")
    
    def _convert_pill_event_msg(self, pill_msg):
        """Convert a PillEvent message to a dictionary"""
        return {
            'time': self._get_relative_time(pill_msg.event_time),
            'tilt': pill_msg.tilt,
            'roll': pill_msg.roll,
            'pitch': pill_msg.pitch,
            'yaw': pill_msg.yaw,
            'pressure': pill_msg.pressure,
            'light': pill_msg.light,
            'pressure_time': self._get_relative_time(pill_msg.pressure_time),
            'light_time': self._get_relative_time(pill_msg.light_time),
            'pick_time': self._get_relative_time(pill_msg.pick_time),
            'place_time': self._get_relative_time(pill_msg.place_time),
            'confirmed': pill_msg.confirmed,
            'forgotten': pill_msg.forgotten
        }
    
    def _get_relative_time(self, ros_time):
        """Convert ROS time to relative time in seconds"""
        return ros_time.to_sec() - self.start_time
    
    def _setup_visualization(self):
        """Setup figures for visualization"""
        plt.ion()
        fig = plt.figure(figsize=(14, 10))
        
        ax1 = fig.add_subplot(511)  # Acceleration
        ax2 = fig.add_subplot(512)  # Orientation
        ax3 = fig.add_subplot(513)  # Pressure
        ax4 = fig.add_subplot(514)  # Light
        ax5 = fig.add_subplot(515)  # Event Timeline
        
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
        ax4.grid(True)
        
        ax5.set_title('Event Timeline')
        ax5.set_ylabel('Event Type')
        ax5.set_xlabel('Time (s)')
        ax5.grid(True)
        
        plt.tight_layout()
        return fig, (ax1, ax2, ax3, ax4, ax5)
    
    def _update_plots(self, frame):
        """Update the visualization plots with latest data"""
        ax1, ax2, ax3, ax4, ax5 = self.axes
        
        for ax in self.axes:
            ax.clear()
        
        # Get current time and set x-axis limits for all plots
        if self.imu_time:
            current_time = max(self.imu_time)
            x_min = max(0, current_time - self.max_time_window)
            
            for ax in self.axes:
                ax.set_xlim(x_min, current_time + 2)
        
        # Plot 1: Acceleration
        if self.imu_time:
            ax1.plot(list(self.imu_time), list(self.ax), 'r-', label="ax", alpha=0.3)
            ax1.plot(list(self.imu_time), list(self.ay), 'g-', label="ay", alpha=0.3)
            ax1.plot(list(self.imu_time), list(self.az), 'b-', label="az", alpha=0.3)
            
            if self.accel_mag:
                ax1.plot(list(self.imu_time)[-len(self.accel_mag):], list(self.accel_mag), 'k-', 
                       label="Accel Magnitude", alpha=0.5)
            
            if self.accel_smooth:
                ax1.plot(list(self.imu_time)[-len(self.accel_smooth):], list(self.accel_smooth), 'm-', 
                       label="Smoothed", alpha=0.7)
                ax1.axhline(y=self.motion_threshold, color='g', linestyle='--', alpha=0.7, label="Threshold")
        
        ax1.set_title('Acceleration')
        ax1.set_ylabel('Magnitude')
        ax1.grid(True)
        ax1.legend(loc='upper left')
        
        # Plot 2: Orientation
        if self.imu_time:
            ax2.plot(list(self.imu_time), list(self.roll), 'r-', label="Roll", alpha=0.7)
            ax2.plot(list(self.imu_time), list(self.pitch), 'g-', label="Pitch", alpha=0.7)
            ax2.plot(list(self.imu_time), list(self.yaw), 'b-', label="Yaw", alpha=0.7)
            
            if self.tilt_mag:
                ax2.plot(list(self.imu_time)[-len(self.tilt_mag):], list(self.tilt_mag), 'k-', 
                       label="Tilt Mag", alpha=0.5)
            
            if self.tilt_change:
                ax2.plot(list(self.imu_time)[-len(self.tilt_change):], list(self.tilt_change), 'c-', 
                       label="Tilt Change", alpha=0.5)
                ax2.axhline(y=self.tilt_threshold, color='r', linestyle='--', alpha=0.7, label="Threshold")
        
        ax2.set_title('Orientation')
        ax2.set_ylabel('Degrees')
        ax2.grid(True)
        ax2.legend(loc='upper left')
        
        # Plot 3: Pressure
        if self.pl_time:
            ax3.plot(list(self.pl_time), list(self.pressure), 'b-')
            ax3.axhline(y=self.pressure_threshold, color='r', linestyle='--', alpha=0.7, label="Event Threshold")
            ax3.axhline(y=self.pressure_threshold * 1.5, color='g', linestyle='--', alpha=0.7, label="Swallow Threshold")
        
        ax3.set_title('Pressure')
        ax3.set_ylabel('Value')
        ax3.grid(True)
        ax3.legend(loc='upper left')
        
        # Plot 4: Light
        if self.pl_time:
            ax4.plot(list(self.pl_time), list(self.light), 'r-')
            ax4.axhline(y=self.light_threshold, color='g', linestyle='--', alpha=0.7, label="Threshold")
        
        ax4.set_title('Light')
        ax4.set_ylabel('Lux')
        ax4.grid(True)
        ax4.legend(loc='upper left')
        
        # Plot 5: Event Timeline
        ax5.set_title('Event Timeline')
        ax5.set_ylabel('Event Type')
        ax5.set_yticks([1, 2, 3, 4])
        ax5.set_yticklabels(['Pick-Place', 'Pill', 'Swallow', 'Forgotten'])
        ax5.set_ylim(0.5, 4.5)
        
        # Plot pick-place events
        for event in self.pick_place_events:
            pick_time = event['pick_time']
            place_time = event['place_time']
            
            # Check if this pick-place contains a pill event
            has_pill = any(pick_time <= pe['time'] <= place_time for pe in self.pill_events)
            
            # Color based on whether it contains a pill event
            color = 'green' if has_pill else 'yellow'
            ax5.plot([pick_time, place_time], [1, 1], color=color, linewidth=4, alpha=0.7)
        
        # Plot pill events
        for event in self.pill_events:
            time_val = event['time']
            
            if event['confirmed']:
                # Confirmed pill - green circle
                ax5.plot(time_val, 2, 'o', markersize=8, color='green', alpha=0.7)
            elif event['forgotten']:
                # Forgotten pill - red circle
                ax5.plot(time_val, 2, 'o', markersize=8, color='red', alpha=0.7)
            else:
                # Normal pill event - blue circle
                ax5.plot(time_val, 2, 'o', markersize=8, color='blue', alpha=0.7)
        
        # Plot swallow events
        for event in self.swallow_events:
            swallow_time = event['swallow_time']
            pill_time = event['pill_event']['time']
            
            # Green marker for swallow
            ax5.plot(swallow_time, 3, 's', markersize=8, color='green', alpha=0.7)
            
            # Connect pill to swallow with line
            ax5.plot([pill_time, swallow_time], [2, 3], 'g-', alpha=0.5)
        
        # Plot forget events
        for event in self.forget_events:
            forget_time = event['forget_time']
            pill_time = event['pill_event']['time']
            
            # Red marker for forget
            ax5.plot(forget_time, 4, 'X', markersize=8, color='red', alpha=0.7)
            
            # Connect pill to forget with line
            ax5.plot([pill_time, forget_time], [2, 4], 'r-', alpha=0.5)
        
        # Add shading for pick-place events
        for event in self.pick_place_events:
            pick_time = event['pick_time']
            place_time = event['place_time']
            
            # Check if this pick-place contains any pill events
            has_pill_event = any(pick_time <= pe['time'] <= place_time for pe in self.pill_events)
            
            # Check if any pills in this pick-place have been forgotten
            has_forgotten_pill = any(
                pick_time <= pe['time'] <= place_time and pe['forgotten'] 
                for pe in self.pill_events
            )
            
            if has_forgotten_pill:
                color = 'red'
                alpha = 0.15
            elif has_pill_event:
                color = 'green'
                alpha = 0.15
            else:
                color = 'yellow'
                alpha = 0.1
            
            for ax in [ax1, ax2, ax3, ax4]:  # Don't shade the event timeline
                y_min, y_max = ax.get_ylim()
                height = y_max - y_min
                rect = Rectangle((pick_time, y_min), width=(place_time - pick_time), height=height, 
                              color=color, alpha=alpha)
                ax.add_patch(rect)
        
        # Update the layout
        plt.tight_layout()
        return self.axes
    
    def run(self):
        """Run the visualizer"""
        plt.show()
        rospy.spin()

if __name__ == "__main__":
    try:
        visualizer = PillDetectionVisualizer()
        visualizer.run()
    except rospy.ROSInterruptException:
        pass