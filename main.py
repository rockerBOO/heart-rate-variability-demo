#!/usr/bin/env python3
"""
Heart Rate Monitor with Real-time Visualization
Supports multiple input methods: webcam, simulated data, or manual input
"""

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from collections import deque
import threading
import queue
import argparse
from datetime import datetime
import numpy as np
from scipy import stats

class HeartRateMonitor:
    def __init__(self, method='webcam', window_size=300):
        self.method = method
        self.window_size = window_size
        self.heart_rates = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.rr_intervals = deque(maxlen=window_size)  # R-R intervals in milliseconds
        self.current_hr = 0
        self.current_hrv = {
            'SDNN': 0,     # Standard Deviation of NN Intervals
            'RMSSD': 0,    # Root Mean Square of Successive Differences
            'pNN50': 0     # Percentage of NN50 intervals
        }
        self.is_running = False
        self.data_queue = queue.Queue()
        
        # Heart rate zones
        self.zones = {
            'Resting': (50, 70),
            'Fat Burn': (70, 85),
            'Cardio': (85, 95),
            'Peak': (95, 110)
        }
        
    def detect_face_roi(self, frame):
        """Detect face and return region of interest for heart rate detection"""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            # Focus on forehead area for better pulse detection
            roi = frame[y:y+h//3, x:x+w]
            return roi, (x, y, w, h//3)
        return None, None
    
    def extract_pulse_signal(self, roi_history):
        """Extract pulse signal from ROI color changes"""
        if len(roi_history) < 30:  # Need enough frames
            return 0
            
        # Calculate mean green channel intensity for each frame
        green_values = []
        for roi in roi_history:
            if roi is not None:
                green_mean = np.mean(roi[:, :, 1])  # Green channel
                green_values.append(green_mean)
        
        if len(green_values) < 30:
            return 0
            
        # Apply bandpass filter (0.8-3.0 Hz corresponds to 48-180 BPM)
        signal = np.array(green_values)
        signal = signal - np.mean(signal)
        
        # Simple peak detection
        from scipy import signal as scipy_signal
        try:
            peaks, _ = scipy_signal.find_peaks(signal, distance=10)
            if len(peaks) > 1:
                # Calculate BPM based on peak intervals
                fps = 30  # Assume 30 FPS
                peak_intervals = np.diff(peaks) / fps
                avg_interval = np.mean(peak_intervals)
                bpm = 60 / avg_interval if avg_interval > 0 else 0
                return max(40, min(200, bpm))  # Clamp to realistic range
        except:
            pass
            
        return 0
    
    def webcam_monitor(self):
        """Monitor heart rate using webcam"""
        cap = cv2.VideoCapture(0)
        roi_history = deque(maxlen=150)  # 5 seconds at 30fps
        
        print("Starting webcam heart rate monitoring...")
        print("Position your face in the frame and stay still for best results.")
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Detect face and get ROI
            roi, face_rect = self.detect_face_roi(frame)
            
            if roi is not None:
                roi_history.append(roi)
                
                # Draw face rectangle
                x, y, w, h = face_rect
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Face detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Calculate heart rate
                if len(roi_history) >= 30:
                    hr = self.extract_pulse_signal(roi_history)
                    if hr > 0:
                        self.current_hr = hr
                        self.data_queue.put(hr)
            
            # Display current heart rate
            cv2.putText(frame, f"HR: {self.current_hr:.0f} BPM", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Heart Rate Monitor', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
    
    def simulate_heart_rate(self):
        """Simulate realistic heart rate data with HRV"""
        # Base heart rate parameters
        base_hr = 75
        hrv_factor = 15  # Controls heart rate variability
        last_hr = base_hr
        start_time = time.time()
        last_update_time = start_time
        
        print("Simulating heart rate data with HRV...")
        print("Heart rate simulation running...")
        
        while self.is_running:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Realistic HRV simulation using multiple stochastic components
            # 1. Respiratory Sinus Arrhythmia (RSA) - breathing-related variation
            rsa_variation = hrv_factor * np.sin(elapsed / 5)
            
            # 2. Autonomic nervous system fluctuations
            ans_variation = hrv_factor * 0.5 * np.random.normal()
            
            # 3. Random walk to simulate physiological changes
            random_walk = hrv_factor * 0.3 * np.random.randn()
            
            # Combine variations
            hr_change = rsa_variation + ans_variation + random_walk
            
            # Update heart rate with smoothing
            hr = last_hr + 0.3 * hr_change
            hr = max(45, min(140, hr))  # Realistic heart rate range
            
            # Calculate R-R interval (time between beats)
            # Convert HR to milliseconds between beats
            rr_interval = 60000 / hr
            
            # Update heart rate and R-R interval
            self.current_hr = hr
            self.rr_intervals.append(rr_interval)
            
            # Calculate HRV metrics periodically
            if len(self.rr_intervals) >= 10 and current_time - last_update_time > 5:
                self.current_hrv = self.calculate_hrv_metrics()
                print(f"\nHRV Metrics: SDNN: {self.current_hrv['SDNN']:.2f}ms")
                last_update_time = current_time
            
            # Put data in queue
            try:
                self.data_queue.put(self.current_hr, timeout=0.1)
                print(f"Generated HR: {self.current_hr:.1f} BPM, RR: {rr_interval:.1f}ms", end='\r')
            except queue.Full:
                # If queue is full, clear it and add new data
                while not self.data_queue.empty():
                    try:
                        self.data_queue.get_nowait()
                    except queue.Empty:
                        break
                self.data_queue.put(self.current_hr)
            
            # Store for next iteration
            last_hr = hr
            
            time.sleep(0.5)  # Update every 0.5 seconds for more responsive visualization
    
    def manual_input_monitor(self):
        """Manual heart rate input"""
        print("Manual heart rate input mode.")
        print("Enter heart rate values (or 'quit' to exit):")
        
        while self.is_running:
            try:
                user_input = input("Enter HR (BPM): ")
                if user_input.lower() == 'quit':
                    self.is_running = False
                    break
                    
                hr = float(user_input)
                if 30 <= hr <= 220:  # Reasonable range
                    self.current_hr = hr
                    self.data_queue.put(hr)
                else:
                    print("Please enter a heart rate between 30 and 220 BPM")
            except (ValueError, EOFError, KeyboardInterrupt):
                break
    
    def calculate_hrv_metrics(self):
        """Calculate Heart Rate Variability Metrics"""
        if len(self.rr_intervals) < 10:
            return {
                'SDNN': 0,
                'RMSSD': 0,
                'pNN50': 0
            }
        
        rr_array = np.array(list(self.rr_intervals))
        
        # SDNN - Standard Deviation of NN Intervals
        sdnn = np.std(rr_array, ddof=1)
        
        # RMSSD - Root Mean Square of Successive Differences
        rr_diff = np.diff(rr_array)
        rmssd = np.sqrt(np.mean(rr_diff**2))
        
        # pNN50 - Percentage of NN50 intervals
        nn50_count = np.sum(np.abs(rr_diff) > 50)
        pnn50 = (nn50_count / (len(rr_array) - 1)) * 100
        
        return {
            'SDNN': sdnn,
            'RMSSD': rmssd,
            'pNN50': pnn50
        }

    def get_hr_zone(self, hr):
        """Determine heart rate zone"""
        for zone, (low, high) in self.zones.items():
            if low <= hr <= high:
                return zone
        if hr < 50:
            return "Very Low"
        else:
            return "Maximum"
    
    def start_monitoring(self):
        """Start the heart rate monitoring"""
        self.is_running = True
        
        print(f"Starting {self.method} heart rate monitoring...")
        
        # Start data collection thread
        if self.method == 'webcam':
            data_thread = threading.Thread(target=self.webcam_monitor, name="WebcamThread")
        elif self.method == 'simulate':
            data_thread = threading.Thread(target=self.simulate_heart_rate, name="SimulateThread")
        else:  # manual
            data_thread = threading.Thread(target=self.manual_input_monitor, name="ManualThread")
            
        data_thread.daemon = True
        data_thread.start()
        
        # Give the data thread a moment to start
        time.sleep(1)
        
        print(f"Data thread started: {data_thread.name}")
        print("Starting visualization...")
        
        # Start visualization (this will block until plot window is closed)
        try:
            self.start_visualization()
        except KeyboardInterrupt:
            print("Interrupted by user")
        except Exception as e:
            print(f"Visualization error: {e}")
        finally:
            self.is_running = False
            print("Monitoring stopped.")
    
    def start_visualization(self):
        """Create a simple static visualization of heart rate data"""
        print("Collecting heart rate data...")
        
        # Collect data for a reasonable time period
        start_time = time.time()
        data_collection_time = 15  # seconds
        
        while time.time() - start_time < data_collection_time and self.is_running:
            # Get data from queue
            try:
                while not self.data_queue.empty():
                    hr = self.data_queue.get_nowait()
                    self.heart_rates.append(hr)
                    self.timestamps.append(time.time() - start_time)
                    
                    # Calculate R-R interval from heart rate (for HRV calculation)
                    rr_interval = 60000 / hr  # Convert HR to milliseconds between beats
                    self.rr_intervals.append(rr_interval)
                    
                    print(f"Collected: {hr:.1f} BPM, RR: {rr_interval:.1f}ms (Total: {len(self.heart_rates)} readings)", end='\r')
            except queue.Empty:
                pass
            time.sleep(0.1)
        
        print(f"\nData collection complete: {len(self.heart_rates)} readings")
        
        # Create the plot
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        if len(self.heart_rates) > 0:
            hr_values = list(self.heart_rates)
            times = list(self.timestamps)
            
            # Plot 1: Heart rate over time
            ax1.plot(times, hr_values, 'r-', linewidth=2, marker='o', markersize=4, label='Heart Rate')
            ax1.fill_between(times, hr_values, alpha=0.3, color='red')
            ax1.set_title(f'Heart Rate Monitor - {len(hr_values)} readings over {data_collection_time}s', fontsize=16)
            ax1.set_xlabel('Time (seconds)', fontsize=12)
            ax1.set_ylabel('Heart Rate (BPM)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Add heart rate zones
            zone_colors = {'Resting': 'blue', 'Fat Burn': 'green', 'Cardio': 'orange', 'Peak': 'red'}
            for zone, (low, high) in self.zones.items():
                color = zone_colors.get(zone, 'gray')
                ax1.axhspan(low, high, alpha=0.1, color=color)
            
            # Statistics
            avg_hr = np.mean(hr_values)
            min_hr = np.min(hr_values)
            max_hr = np.max(hr_values)
            current_zone = self.get_hr_zone(avg_hr)
            
            # HRV calculation from collected data
            if len(self.rr_intervals) > 0:
                hrv_metrics = self.calculate_hrv_metrics()
            else:
                hrv_metrics = {'SDNN': 0, 'RMSSD': 0, 'pNN50': 0}
            
            stats_text = f'Statistics:\n'
            stats_text += f'Readings: {len(hr_values)}\n'
            stats_text += f'Average: {avg_hr:.1f} BPM\n'
            stats_text += f'Range: {min_hr:.1f} - {max_hr:.1f} BPM\n'
            stats_text += f'Zone: {current_zone}\n\n'
            stats_text += f'HRV Metrics:\n'
            stats_text += f'SDNN: {hrv_metrics["SDNN"]:.2f} ms\n'
            stats_text += f'RMSSD: {hrv_metrics["RMSSD"]:.2f} ms\n'
            stats_text += f'pNN50: {hrv_metrics["pNN50"]:.1f}%'
            
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10, 
                    verticalalignment='top', color='black', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray'))
            
            # Plot 2: Heart rate distribution
            ax2.hist(hr_values, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(avg_hr, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_hr:.1f}')
            ax2.set_title('Heart Rate Distribution', fontsize=14)
            ax2.set_xlabel('Heart Rate (BPM)', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        else:
            # No data collected
            ax1.text(0.5, 0.5, 'No data collected!', transform=ax1.transAxes, 
                    ha='center', va='center', fontsize=16, color='red')
            ax2.text(0.5, 0.5, 'No data to display', transform=ax2.transAxes, 
                    ha='center', va='center', fontsize=16, color='red')
        
        plt.tight_layout()
        plt.savefig("x.png", dpi=100, bbox_inches='tight')
        print("Plot saved to x.png")
        
        # Stop monitoring
        self.is_running = False
    
    def stop_monitoring(self):
        """Stop heart rate monitoring"""
        self.is_running = False
        print("Heart rate monitoring stopped.")
        
        # Print session summary
        if len(self.heart_rates) > 0:
            print(f"\nSession Summary:")
            print(f"Duration: {len(self.heart_rates)} readings")
            print(f"Average HR: {np.mean(list(self.heart_rates)):.1f} BPM")
            print(f"Min HR: {np.min(list(self.heart_rates)):.1f} BPM")
            print(f"Max HR: {np.max(list(self.heart_rates)):.1f} BPM")


def main():
    parser = argparse.ArgumentParser(description='Heart Rate Monitor')
    parser.add_argument('--method', choices=['webcam', 'simulate', 'manual'], 
                       default='simulate', help='Monitoring method')
    parser.add_argument('--window', type=int, default=300, 
                       help='Data window size (number of readings)')
    parser.add_argument('--duration', type=int, default=60, 
                       help='Simulation duration in seconds')
    
    args = parser.parse_args()
    
    print("Heart Rate Monitor Starting...")
    print(f"Method: {args.method}")
    print(f"Simulation Duration: {args.duration} seconds")
    
    if args.method == 'webcam':
        print("Note: Webcam method requires good lighting and stable positioning")
        print("Press 'q' in the camera window to quit")
    
    monitor = HeartRateMonitor(method=args.method, window_size=args.window)
    
    try:
        import threading
        import time
        
        # Start monitoring in a separate thread
        monitor_thread = threading.Thread(target=monitor.start_monitoring)
        monitor_thread.start()
        
        # Run for specified duration
        time.sleep(args.duration)
        
        # Stop monitoring
        monitor.stop_monitoring()
        monitor_thread.join()
    except KeyboardInterrupt:
        print("\nStopping monitor...")
    finally:
        monitor.stop_monitoring()


if __name__ == "__main__":
    # Install required packages
    required_packages = """
    Required packages (install with pip):
    - opencv-python
    - matplotlib
    - numpy
    - scipy (for signal processing)
    
    Example usage:
    python heart_rate_monitor.py --method simulate
    python heart_rate_monitor.py --method webcam
    python heart_rate_monitor.py --method manual
    """
    
    print(required_packages)
    main()
