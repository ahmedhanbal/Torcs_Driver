import sys
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import datetime
import os
import socket
import time

# Import modules from pyScrcClient
sys.path.insert(0, 'pyScrcClient/src')
from driver import Driver
from msgParser import MsgParser
from carState import CarState

# Create a SimpleParser class as a wrapper for msgParser
class SimpleParser:
    """
    A simple parser for TORCS data that provides easier access to parsed data.
    """
    def __init__(self):
        self.msg_parser = MsgParser()
        self.car_state = CarState()
        
        # Initialize attributes for sensor data
        self.angle = 0.0
        self.track_position = 0.0
        self.speed_x = 0.0
        self.speed_y = 0.0
        self.speed_z = 0.0
        self.track_edge_sensors = [0.0] * 19  # 19 track sensors
        self.wheel_velocities = [0.0] * 4     # 4 wheels
        self.opponents = [200.0] * 36         # 36 opponent sensors
        self.rpm = 0.0
        self.gear = 0
        self.damage = 0.0
        self.z = 0.0
        self.fuel = 0.0
        self.current_lap_time = 0.0
        self.last_lap_time = 0.0
        self.distance_from_start = 0.0
        self.distance_covered = 0.0
        self.race_position = 0
        self.track_name = None
        self.car_name = None
    
    def parse(self, buffer):
        """
        Parse the buffer from TORCS server and extract sensor data.
        
        Args:
            buffer: String containing data from TORCS server
        """
        try:
            # Handle possible issues with the buffer
            if not buffer or not isinstance(buffer, str):
                print(f"Invalid buffer: {type(buffer)}")
                return
                
            # Parse data
            sensor_dict = self.msg_parser.parse(buffer)
            if sensor_dict is None:
                print("No sensor data parsed")
                return
            
            # Update car state using the parsed sensor data
            self.car_state.setFromMsg(sensor_dict)
            
            # Extract and store relevant sensor values for easy access
            try:
                # Angle and positions
                self.angle = self.car_state.getAngle()
                self.track_position = self.car_state.getTrackPos()
                
                # Speed
                self.speed_x = self.car_state.getSpeedX()
                self.speed_y = self.car_state.getSpeedY()
                self.speed_z = self.car_state.getSpeedZ()
                
                # Track sensors
                sensors = self.car_state.getTrack()
                if sensors:
                    self.track_edge_sensors = sensors
                
                # Wheel velocities
                wheel_spin = self.car_state.getWheelSpinVel()
                if wheel_spin:
                    self.wheel_velocities = wheel_spin
                
                # Opponents
                opponents = self.car_state.getOpponents()
                if opponents:
                    self.opponents = opponents
                
                # Other sensors
                self.rpm = self.car_state.getRpm()
                self.gear = self.car_state.getGear()
                self.damage = self.car_state.getDamage()
                self.z = self.car_state.getZ()
                self.fuel = self.car_state.getFuel()
                
                # Race information
                self.current_lap_time = self.car_state.getCurLapTime()
                self.last_lap_time = self.car_state.getLastLapTime()
                self.distance_from_start = self.car_state.getDistFromStart()
                self.distance_covered = self.car_state.getDistRaced()
                self.race_position = self.car_state.getRacePos()
                
                # Try to get track and car name from dictionary if available
                if 'trackname' in sensor_dict and sensor_dict['trackname']:
                    self.track_name = sensor_dict['trackname'][0]
                if 'carname' in sensor_dict and sensor_dict['carname']:
                    self.car_name = sensor_dict['carname'][0]
            except Exception as e:
                print(f"Error processing sensor data: {e}")
                
        except Exception as e:
            print(f"Parser error: {e}")

# Create our own Client class based on client.py functionality
class Client:
    """Base TORCS client class for handling network communication."""
    
    def __init__(self, driver_instance):
        """Initialize the client with a driver instance."""
        self.driver = driver_instance
        self.parser = SimpleParser()
        self.socket = None
        self.host = 'localhost'
        self.port = 3001
        self.id = 'SCR'
        self.shutdown_requested = False
    
    def run(self, host='localhost', port=3001, client_id='SCR'):
        """Run the client, connecting to the server and processing messages."""
        self.host = host
        self.port = port
        self.id = client_id
        
        # Initialize socket
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.settimeout(1.0)
        except socket.error as msg:
            print('Could not make a socket:', msg)
            sys.exit(-1)
        
        # Identification loop
        while True:
            print(f'Sending id to server: {self.id}')
            buf = self.id + self.driver.init()
            
            try:
                self.socket.sendto(buf.encode('utf-8'), (self.host, self.port))
            except socket.error as msg:
                print("Failed to send data...Exiting...")
                sys.exit(-1)
                
            try:
                buf, addr = self.socket.recvfrom(1000)
                buf = buf.decode('utf-8')
            except socket.error as msg:
                print("didn't get response from server...")
                continue
        
            if buf.find('***identified***') >= 0:
                print('Received:', buf)
                break
        
        # Main simulation loop
        while not self.shutdown_requested:
            # Wait for server response
            buf = None
            try:
                buf, addr = self.socket.recvfrom(1000)
                buf = buf.decode('utf-8')
            except socket.error as msg:
                print("didn't get response from server...")
                continue  # Skip this iteration and try again
            
            try:
                # Process server message - check special messages before parsing
                if buf:
                    # Debug output
                    if self.driver.debug:
                        print(f"Received data: {buf[:50]}...")  # Print first 50 chars
                    
                    # Check for shutdown/restart messages before parsing
                    if buf.find('***shutdown***') >= 0:
                        self.shutdown()
                        print('Client Shutdown')
                        break
                    
                    if buf.find('***restart***') >= 0:
                        # Restart the race
                        self.driver.onRestart()
                        print('Client Restart')
                        break
                    
                    # Normal step - parse data and drive
                    self.parser.parse(buf)
                    control = self.driver.drive(self, self.parser)
                    
                    if control is not None:
                        self.socket.sendto(control.encode('utf-8'), (self.host, self.port))
            except Exception as e:
                print(f"Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                # Continue rather than crashing on a single error
                continue
    
    def shutdown(self):
        """Shutdown the client."""
        self.shutdown_requested = True
        if self.socket:
            self.socket.close()

class NeuralDriver(Driver):
    """
    Neural network-based driver for TORCS.
    This class uses a trained model to predict control actions.
    """
    
    def __init__(self, model_path='model/torcs_model', stage=0, **kwargs):
        # Extract kwargs that should not be passed to the parent Driver class
        self.enable_logging = kwargs.pop('enable_logging', False)
        self.debug = kwargs.pop('debug', False)
        
        # Get car and track info if available
        car_name = kwargs.pop('car_name', "ToyotaCorollaWRC")
        track_name = kwargs.pop('track_name', "unknown")
        mode = kwargs.pop('mode', "ai")  # Always use AI mode for neural driver
        
        # Call parent constructor with only the arguments it accepts
        super().__init__(stage=stage, mode=mode, car_name=car_name, track_name=track_name)
        
        # Model path handling - support both directory and file path formats
        if os.path.isdir(model_path):
            model_file = os.path.join(model_path, "torcs_model.h5")
            scaler_file = os.path.join(model_path, "torcs_model_scaler.pkl")
            input_cols_file = os.path.join(model_path, "torcs_model_input_cols.npy")
            output_cols_file = os.path.join(model_path, "torcs_model_output_cols.npy")
        else:
            model_file = f"{model_path}.h5"
            scaler_file = f"{model_path}_scaler.pkl"
            input_cols_file = f"{model_path}_input_cols.npy"
            output_cols_file = f"{model_path}_output_cols.npy"
        
        # Load model components
        try:
            self.model = load_model(model_file)
            self.scaler = joblib.load(scaler_file)
            self.input_cols = np.load(input_cols_file, allow_pickle=True)
            self.output_cols = np.load(output_cols_file, allow_pickle=True)
            print(f"Loaded model with {len(self.input_cols)} input features and {len(self.output_cols)} outputs")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.scaler = None
            self.input_cols = []
            self.output_cols = []
        
        # Initialize sensor data dict
        self.sensor_data = {}
        
        # Track state initialization
        self.track_name = track_name
        self.car_name = car_name
        
        # For logging
        self.log_file = None
        self.log_writer = None
        
        # Statistics
        self.prediction_count = 0
        self.total_prediction_time = 0
    
    def drive(self, client, parser):
        """
        Drive method called by the TORCS client.
        
        Args:
            client: TORCS client instance
            parser: TORCS parser instance
            
        Returns:
            Control commands for the car
        """
        import time
        
        # Default control values
        self.steer = 0.0
        self.accelerate = 0.0
        self.brake = 0.0
        self.gear = 1
        self.clutch = 0.0
        
        # Check if model is loaded
        if self.model is None or self.scaler is None:
            print("Model not loaded properly. Using default controls.")
            return self.command()
        
        try:
            # Get the current sensor data
            sensors = self.process_sensors(parser)
            
            # Track some metadata if available
            if parser.track_name and not self.track_name:
                self.track_name = parser.track_name
            if parser.car_name and not self.car_name:
                self.car_name = parser.car_name
            
            # Prepare input data for the model
            input_data = {}
            for col in self.input_cols:
                if col in sensors:
                    input_data[col] = sensors[col]
                else:
                    # Handle missing features - use 0 as default
                    if self.debug:
                        print(f"Warning: Missing feature {col}, using 0")
                    input_data[col] = 0
            
            # Convert to DataFrame and then to numpy array
            input_df = pd.DataFrame([input_data])
            
            # Apply scaling
            start_time = time.time()
            X = self.scaler.transform(input_df)
            
            # Get model prediction
            prediction = self.model.predict(X, verbose=0)[0]
            prediction_time = time.time() - start_time
            
            # Update statistics
            self.prediction_count += 1
            self.total_prediction_time += prediction_time
            
            # Extract control outputs
            controls = {}
            for i, col in enumerate(self.output_cols):
                controls[col] = prediction[i]
            
            # Apply controls to the car
            self.steer = controls.get('Steering', 0.0)
            self.accelerate = controls.get('Acceleration', 0.0)
            self.brake = controls.get('Braking', 0.0)
            
            # Handle gear changes 
            if 'Gear' in controls:
                self.gear = int(round(controls.get('Gear')))
            
            # Handle clutch
            if 'Clutch' in controls:
                self.clutch = controls.get('Clutch', 0.0)
            
            # Log data if enabled
            if self.enable_logging and self.log_writer:
                log_data = {**sensors, **controls}
                self.log_writer.writerow(log_data)
            
            if self.debug and self.prediction_count % 100 == 0:
                avg_time = self.total_prediction_time / self.prediction_count
                print(f"Prediction {self.prediction_count}, "
                      f"Avg time: {avg_time*1000:.2f}ms, "
                      f"Steer: {self.steer:.3f}, "
                      f"Accel: {self.accelerate:.3f}, "
                      f"Brake: {self.brake:.3f}, "
                      f"Gear: {self.gear}")
            
        except Exception as e:
            print(f"Error in drive method: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
        
        return self.command()
    
    def process_sensors(self, parser):
        """
        Process sensor data from TORCS parser.
        
        Args:
            parser: TORCS parser instance
            
        Returns:
            Dictionary of processed sensor data
        """
        # Extract all sensor data from parser
        sensors = {}
        
        try:
            # Add angle, track position and speed
            sensors['Angle'] = getattr(parser, 'angle', 0.0)
            sensors['TrackPosition'] = getattr(parser, 'track_position', 0.0)
            sensors['SpeedX'] = getattr(parser, 'speed_x', 0.0)
            sensors['SpeedY'] = getattr(parser, 'speed_y', 0.0)
            sensors['SpeedZ'] = getattr(parser, 'speed_z', 0.0)
            
            # Add distances to track edges
            for i, track_edge in enumerate(getattr(parser, 'track_edge_sensors', [0.0] * 19)):
                if i < 19:  # Ensure we don't exceed the expected number
                    sensors[f'Track_{i+1}'] = track_edge
            
            # Add wheel spin velocity
            for i, wheel_spin_vel in enumerate(getattr(parser, 'wheel_velocities', [0.0] * 4)):
                if i < 4:  # Ensure we don't exceed the expected number
                    sensors[f'WheelSpinVelocity_{i+1}'] = wheel_spin_vel
            
            # Add opponent sensors - with default values to prevent errors
            for i, opponent in enumerate(getattr(parser, 'opponents', [200.0] * 36)):
                if i < 36:  # Ensure we don't exceed the expected number
                    sensors[f'Opponent_{i+1}'] = opponent
            
            # Add other sensor data - all with safe defaults
            sensors['RPM'] = getattr(parser, 'rpm', 0.0)
            sensors['Gear'] = getattr(parser, 'gear', 0)
            sensors['Damage'] = getattr(parser, 'damage', 0.0)
            sensors['Z'] = getattr(parser, 'z', 0.0)
            sensors['FuelLevel'] = getattr(parser, 'fuel', 0.0)
            sensors['CurrentLapTime'] = getattr(parser, 'current_lap_time', 0.0)
            sensors['LastLapTime'] = getattr(parser, 'last_lap_time', 0.0)
            sensors['DistanceFromStart'] = getattr(parser, 'distance_from_start', 0.0)
            sensors['DistanceCovered'] = getattr(parser, 'distance_covered', 0.0)
            sensors['RacePosition'] = getattr(parser, 'race_position', 0)
            
            if self.debug:
                print(f"Processed {len(sensors)} sensor values")
        except Exception as e:
            print(f"Error extracting sensor data: {e}")
            # Continue with whatever data was collected
        
        return sensors
    
    def start_logging(self, output_dir='data'):
        """
        Start logging sensor and control data to a CSV file.
        
        Args:
            output_dir: Directory to save log files
        """
        import csv
        
        if not self.enable_logging:
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a subdirectory for the track if known
        track_dir = self.track_name if self.track_name else "unknown_track"
        track_dir = os.path.join(output_dir, track_dir)
        os.makedirs(track_dir, exist_ok=True)
        
        # Generate a filename based on track, car, mode and timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        car_name = self.car_name if self.car_name else "unknown_car"
        filename = f"{self.track_name}_{car_name}_ai_{timestamp}.csv"
        filepath = os.path.join(track_dir, filename)
        
        print(f"Logging data to {filepath}")
        
        # Create a CSV writer for logging
        self.log_file = open(filepath, 'w', newline='')
        
        # Determine all possible columns (inputs + outputs)
        all_columns = list(self.input_cols) + list(self.output_cols)
        
        self.log_writer = csv.DictWriter(self.log_file, fieldnames=all_columns)
        self.log_writer.writeheader()
    
    def stop_logging(self):
        """Stop logging and close the log file."""
        if self.log_file:
            self.log_file.close()
            self.log_file = None
            self.log_writer = None

    def command(self):
        """
        Build the command string to send to the TORCS server.
        Similar to the Driver.drive() method's return value.
        
        Returns:
            String containing control commands for TORCS
        """
        from carControl import CarControl
        
        # Create a new CarControl instance
        control = CarControl()
        
        # Set the control values based on neural network output
        control.setSteer(self.steer)
        control.setAccel(self.accelerate)
        control.setBrake(self.brake)
        control.setClutch(self.clutch)
        control.setGear(self.gear)
        
        # Get the string representation for the server
        return control.toMsg()

class NeuralClient(Client):
    """TORCS client using neural network for control."""
    
    def __init__(self, model_path='model/torcs_model', **kwargs):
        self.driver = NeuralDriver(model_path=model_path, **kwargs)
        super().__init__(self.driver)
    
    def shutdown(self):
        """Shutdown the client and stop logging."""
        self.driver.stop_logging()
        super().shutdown()

def main():
    """Main function to run the neural network client."""
    import argparse
    
    parser = argparse.ArgumentParser(description='TORCS Neural Network Client')
    parser.add_argument('--model', type=str, default='model/torcs_model', 
                        help='Path to the trained model (without .h5 extension)')
    parser.add_argument('--host', type=str, default='localhost', 
                        help='TORCS server host')
    parser.add_argument('--port', type=int, default=3001, 
                        help='TORCS server port')
    parser.add_argument('--logging', action='store_true', 
                        help='Enable logging of sensor and control data')
    parser.add_argument('--debug', action='store_true', 
                        help='Enable debug output')
    
    args = parser.parse_args()
    
    # Check dependencies
    missing_packages = []
    for package, module in [
        ('tensorflow', 'tensorflow'), 
        ('joblib', 'joblib'),
        ('pandas', 'pandas'), 
        ('numpy', 'numpy')
    ]:
        try:
            __import__(module)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print(f"Install with: pip install {' '.join(missing_packages)}")
        print("Continuing anyway, but functionality may be limited.")
    
    try:
        client = NeuralClient(
            model_path=args.model,
            enable_logging=args.logging,
            debug=args.debug
        )
        
        # Start logging if enabled
        if args.logging:
            client.driver.start_logging()
        
        # Connect to the TORCS server
        client.run(host=args.host, port=args.port)
    
    except KeyboardInterrupt:
        print("Client interrupted by user. Shutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'client' in locals():
            client.shutdown()

if __name__ == "__main__":
    main() 