import os
import csv
import datetime
import time
import json

class Logger:
    """
    Logger class for TORCS client to log driving data in CSV format.
    """
    
    def __init__(self):
        """Constructor"""
        self.sensor_data = []
        self.control_data = []
        self.base_dir = os.path.join("..", "pyScrcClient", "data")  # Base directory for data
        self.current_file = None
        self.writer = None
        self.session_data = {}
        self.session_file = os.path.join("..", "torcs_session_data.json")
        self._load_session_data()
        
    def _load_session_data(self):
        """Load existing session data from JSON file if it exists"""
        try:
            if os.path.exists(self.session_file):
                with open(self.session_file, 'r') as f:
                    self.session_data = json.load(f)
            else:
                self.session_data = {
                    "runs": {},
                    "last_run_id": 0,
                    "total_sessions": 0
                }
        except Exception as e:
            print(f"Warning: Failed to load session data: {e}")
            self.session_data = {
                "runs": {},
                "last_run_id": 0,
                "total_sessions": 0
            }
            
    def _save_session_data(self):
        """Save session data to JSON file"""
        try:
            with open(self.session_file, 'w') as f:
                json.dump(self.session_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save session data: {e}")
        
    def get_next_run_id(self, track_name, car_name, mode):
        """Get the next run ID for this car/track/mode combination"""
        key = f"{track_name}_{car_name}_{mode}"
        
        # If we have this combination already, increment its counter
        if key in self.session_data["runs"]:
            self.session_data["runs"][key] += 1
        else:
            # First run for this combination
            self.session_data["runs"][key] = 1
            
        # Increment global counter
        self.session_data["last_run_id"] += 1
        self.session_data["total_sessions"] += 1
        
        # Save updated session data
        self._save_session_data()
        
        return self.session_data["runs"][key]
        
    def initialize_log(self, track_name, car_name, mode, run_id=None):
        """Initialize a new log file with the specified naming convention"""
        # Create track-specific directory path
        track_dir = os.path.join(self.base_dir, track_name)
        
        # Ensure track directory exists
        if not os.path.exists(track_dir):
            os.makedirs(track_dir)
            
        # Generate run_id if not provided
        if run_id is None:
            run_id = self.get_next_run_id(track_name, car_name, mode)
        
        # Filename format: [track]_[car]_[mode]_[run_id].csv
        original_run_id = run_id
        
        # Check if file already exists and increment run_id or add timestamp to avoid overwriting
        filename = f"{track_name}_{car_name}_{mode}_{run_id:02d}.csv"
        filepath = os.path.join(track_dir, filename)
        
        # If file exists, try incrementing the run_id or adding timestamp
        if os.path.exists(filepath):
            # First try incrementing the run_id
            attempt = 1
            max_attempts = 100
            
            while os.path.exists(filepath) and attempt < max_attempts:
                run_id = original_run_id + attempt
                filename = f"{track_name}_{car_name}_{mode}_{run_id:02d}.csv"
                filepath = os.path.join(track_dir, filename)
                attempt += 1
                
            # If we still have a conflict after incrementing, add a timestamp
            if os.path.exists(filepath):
                timestamp = int(time.time()) % 10000  # Last 4 digits of timestamp
                filename = f"{track_name}_{car_name}_{mode}_{original_run_id:02d}_{timestamp}.csv"
                filepath = os.path.join(track_dir, filename)
        
        self.current_file = filepath
        
        # Update session metadata for this run
        session_key = f"{track_name}_{car_name}_{mode}"
        run_info = {
            "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": filename,
            "file_path": filepath,
            "track": track_name,
            "car": car_name,
            "mode": mode,
            "run_id": run_id,
            "status": "active"
        }
        
        # Add run to session data
        if "run_history" not in self.session_data:
            self.session_data["run_history"] = []
            
        self.session_data["run_history"].append(run_info)
        self.session_data["current_run"] = run_info
        self._save_session_data()
        
        # Create and initialize the CSV file with headers
        with open(self.current_file, 'w', newline='') as csvfile:
            self.writer = csv.writer(csvfile)
            # Header based on standard format
            self.writer.writerow([
                "Angle", "CurrentLapTime", "Damage", "DistanceFromStart", "DistanceCovered", 
                "FuelLevel", "Gear", "LastLapTime", 
                # Opponent sensors (1-36)
                *[f"Opponent_{i}" for i in range(1, 37)],
                "RacePosition", "RPM", "SpeedX", "SpeedY", "SpeedZ", 
                # Track sensors (1-19)
                *[f"Track_{i}" for i in range(1, 20)],
                "TrackPosition", 
                # Wheel spin velocities (1-4)
                *[f"WheelSpinVelocity_{i}" for i in range(1, 5)],
                "Z", 
                # Control outputs
                "Acceleration", "Braking", "Clutch", "Gear", "Steering",
                # Metadata
                "Timestamp", "RunID", "Frame"
            ])
        
        print(f"Started logging to {self.current_file}")
        self.frame_counter = 0
        return self.current_file
    
    def log_data(self, sensor_model, control):
        """Log a single data point with sensor and control data"""
        # Accumulate data in memory
        self.sensor_data.append(sensor_model)
        self.control_data.append(control)
        
        # Write to file immediately
        if self.current_file:
            with open(self.current_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Increment frame counter
                self.frame_counter += 1
                
                # Extract data as needed
                wheel_spin_vel = sensor_model.getWheelSpinVel() or [0, 0, 0, 0]
                if len(wheel_spin_vel) < 4:
                    wheel_spin_vel.extend([0] * (4 - len(wheel_spin_vel)))
                
                # Get track sensors (defaulting to 200.0 when not available)
                track_sensors = sensor_model.getTrack() or []
                while len(track_sensors) < 19:
                    track_sensors.append(200.0)
                
                # Get opponent sensors (defaulting to 200.0 when not available)
                opponents = sensor_model.getOpponents() or []
                while len(opponents) < 36:
                    opponents.append(200.0)
                
                # Current timestamp for data integrity
                timestamp = time.time()
                
                # Write a row in CSV format
                writer.writerow([
                    sensor_model.getAngle() or 0,
                    sensor_model.getCurLapTime() or 0,
                    sensor_model.getDamage() or 0,
                    sensor_model.getDistFromStart() or 0,
                    sensor_model.getDistRaced() or 0,  # DistanceCovered
                    sensor_model.getFuelLevel() or 0,
                    sensor_model.getGear() or 0,
                    sensor_model.getLastLapTime() or 0,
                    # All 36 opponent sensors
                    *opponents,
                    sensor_model.getRacePos() or 0,
                    sensor_model.getRpm() or 0,
                    sensor_model.getSpeedX() or 0,
                    sensor_model.getSpeedY() or 0,
                    sensor_model.getSpeedZ() or 0,
                    # All 19 track sensors
                    *track_sensors,
                    sensor_model.getTrackPos() or 0,
                    # All 4 wheel spin velocities
                    wheel_spin_vel[0],
                    wheel_spin_vel[1],
                    wheel_spin_vel[2],
                    wheel_spin_vel[3],
                    sensor_model.getZ() or 0,
                    # Control outputs
                    control.getAccel() or 0,
                    control.getBrake() or 0,
                    control.getClutch() or 0,
                    control.getGear() or 0,
                    control.getSteer() or 0,
                    # Metadata
                    timestamp,
                    self.session_data.get("current_run", {}).get("run_id", 0),
                    self.frame_counter
                ])
                
    def close(self):
        """Close the current log file"""
        if self.current_file:
            # Update session data with completion status
            if "current_run" in self.session_data:
                self.session_data["current_run"]["status"] = "completed"
                self.session_data["current_run"]["end_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.session_data["current_run"]["frames"] = self.frame_counter
                
                # Find the run in run_history and update it
                for run in self.session_data.get("run_history", []):
                    if run.get("file_path") == self.current_file:
                        run["status"] = "completed"
                        run["end_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        run["frames"] = self.frame_counter
                        break
                
                self._save_session_data()
                
            print(f"Logging completed: {self.current_file} ({self.frame_counter} frames)")
            self.sensor_data = []
            self.control_data = []
            self.current_file = None
            self.frame_counter = 0 