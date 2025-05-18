import msgParser
import carState
import carControl
import keyboardController
import math

class Driver(object):
    '''
    Driver object for TORCS with both AI and manual driving capabilities
    '''

    # Racing stages
    WARM_UP = 0
    QUALIFYING = 1
    RACE = 2
    UNKNOWN = 3
    
    # Driving modes
    MODE_AI = "ai"
    MODE_MANUAL = "manual"
    
    # Supported cars and tracks
    SUPPORTED_CARS = ["ToyotaCorollaWRC", "Peugeot406", "MitsubishiLancer"]
    SUPPORTED_TRACKS = ["G-Speedway", "E-Track3", "Dirt2"]

    def __init__(self, stage=UNKNOWN, mode=MODE_AI, car_name="ToyotaCorollaWRC", track_name="unknown"):
        '''Constructor'''
        self.stage = stage
        self.mode = mode
        self.car_name = car_name
        self.track_name = track_name
        
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        
        # AI driving parameters
        self.steer_lock = 0.785398  # Approx. 45 degrees
        self.max_speed = 100  # m/s
        self.prev_rpm = None
        
        # Enhanced AI driving parameters for different tracks
        self.track_params = {
            "G-Speedway": {  # Oval track
                "max_speed": 150,
                "lookahead_dist": 10,
                "steer_gain": 0.8,
                "speed_gain": 1.2
            },
            "E-Track3": {  # Road track
                "max_speed": 120,
                "lookahead_dist": 10,
                "steer_gain": 1.0,
                "speed_gain": 1.0
            },
            "Dirt2": {  # Dirt track
                "max_speed": 80,
                "lookahead_dist": 4,
                "steer_gain": 1.2,
                "speed_gain": 0.8
            },
        }
        
        # Advanced car-specific parameters from XML files
        self.car_params = {
            "ToyotaCorollaWRC": {  
                "gear_up_rpm": 8500,  # From revs limiter in XML
                "gear_down_rpm": 3500,
                "brake_power": 1.0,
                "accel_power": 1.0,
                "drivetrain": "4WD",
                "steer_lock_rad": 0.436,  # 25 degrees in radians
                "brake_balance": 0.62,    # front-rear brake repartition
                "max_brake_pressure": 15000,
                "gear_ratios": [-2.96, 2.232, 1.625, 1.086, 0.856, 0.69, 0.6],
                "max_torque": 472,        # Maximum torque
                "max_torque_rpm": 5888,   # RPM at which max torque occurs
                "torque_curve": [          # RPM to torque mapping from XML
                    (0, 100), (1024, 227), (2048, 310), (3072, 375),
                    (4096, 471), (5120, 472), (5888, 462), (7168, 438),
                    (8192, 364), (9216, 210), (10000, 157)
                ],
                "abs_slip_threshold": 3.0,
                "tc_slip_threshold": 5.0,
                "weight": 1350,           # kg
                "gear_efficiency": [0.957, 0.955, 0.957, 0.950, 0.983, 0.948, 0.940],
                "tire_mu": 1.5,           # Tire friction coefficient
                "central_diff": {         # From XML for 4WD cars
                    "type": "VISCOUS COUPLER",
                    "min_torque_bias": 0.4,  # 40%
                    "max_torque_bias": 0.6   # 60%
                },
                "diff_type": "LIMITED SLIP",
                "diff_ratio": 6.85,
                "fuel_consumption": 1.1,
                "suspension_travel": 0.4,  # m
                "ground_clearance": 0.23,  # m
                "car_width": 1.98,         # m
                "car_length": 3.81         # m
            },
            "Peugeot406": {  
                "gear_up_rpm": 6500,       # From revs limiter in XML
                "gear_down_rpm": 3000,
                "brake_power": 1.2,        # Better brakes (higher max pressure)
                "accel_power": 0.9,        # Slightly less powerful
                "drivetrain": "FWD",
                "steer_lock_rad": 0.663,   # 38 degrees in radians
                "brake_balance": 0.6,      # front-rear brake repartition
                "max_brake_pressure": 55000, # Higher brake pressure
                "gear_ratios": [-5.0, 3.82, 2.15, 1.56, 1.21, 0.97],
                "max_torque": 260,         # Maximum torque
                "max_torque_rpm": 4500,    # RPM at which max torque occurs
                "torque_curve": [          # RPM to torque mapping from XM
                    (0, 20), (500, 60), (1000, 110), (1500, 110),
                    (2000, 140), (2500, 167), (3000, 195), (3500, 220),
                    (4000, 240), (4500, 260), (5000, 255), (5500, 250),
                    (6000, 220), (6500, 200), (7000, 160)
                ],
                "abs_slip_threshold": 2.5, # More sensitive ABS
                "tc_slip_threshold": 4.0,  # More sensitive traction control
                "weight": 1500,            # kg
                "gear_efficiency": [0.87, 0.89, 0.89, 0.89, 0.90, 0.91],
                "tire_mu": 1.3,            # Lower friction coefficient for road tires
                "central_diff": None,      # No central diff on FWD
                "diff_type": "FREE",       # Front differential type
                "diff_ratio": 3.7,
                "fuel_consumption": 1.08,
                "suspension_travel": 0.4,  # m
                "ground_clearance": 0.33,  # m
                "car_width": 2.0,          # m
                "car_length": 4.64         # m
            },
            "MitsubishiLancer": {  # Evo VI WRC
                "gear_up_rpm": 8200,       # From revs limiter in XML
                "gear_down_rpm": 3200,
                "brake_power": 1.1,
                "accel_power": 1.1,        # More powerful than Corolla
                "drivetrain": "4WD",
                "steer_lock_rad": 0.489,   # 28 degrees in radians
                "brake_balance": 0.65,     # front-rear brake repartition
                "max_brake_pressure": 15000,
                "gear_ratios": [-2.96, 2.232, 1.625, 1.086, 0.856, 0.69, 0.6],
                "max_torque": 472,         # Maximum torque
                "max_torque_rpm": 5888,    # RPM at which max torque occurs
                "torque_curve": [          # RPM to torque mapping from XML
                    (0, 0), (1024, 230), (2048, 345), (3072, 395),
                    (4096, 461), (5120, 471), (5888, 472), (7168, 458),
                    (8192, 354), (9216, 200), (10000, 147)
                ],
                "abs_slip_threshold": 2.8,
                "tc_slip_threshold": 4.5,
                "weight": 1350,            # kg
                "gear_efficiency": [0.957, 0.955, 0.957, 0.950, 0.983, 0.948, 0.940],
                "tire_mu": 1.5,            # Tire friction coefficient
                "central_diff": {          # From XML for 4WD cars
                    "type": "VISCOUS COUPLER",
                    "min_torque_bias": 0.4,  # 40%
                    "max_torque_bias": 0.6   # 60%
                },
                "diff_type": "LIMITED SLIP",
                "diff_ratio": 6.85,
                "fuel_consumption": 1.1,
                "suspension_travel": 0.4,  # m
                "ground_clearance": 0.25,  # m
                "car_width": 2.02,         # m
                "car_length": 4.20         # m
            }
        }
        
        # Initialize parameters based on track and car
        self._init_driver_params()
        
        # Manual driving controller
        self.keyboard = keyboardController.KeyboardController()
        self.keyboard.set_mode_toggle_callback(self._toggle_driving_mode)
        
        # Start keyboard controller if in manual mode
        if self.mode == self.MODE_MANUAL:
            self.keyboard.start()
            
        # For ABS system
        self.prev_wheel_slip = [0, 0, 0, 0]
        
        # For traction control
        self.prev_slip = 0
        
        # To handle car starting direction
        self.startup_counter = 0
        self.initial_accel_phase = True
        
        # For clutch control
        self.clutch_max = 0.5
        self.clutch_delta = 0.05
        self.clutch_delta_time = 0.02
        self.clutch_delta_raced = 10
        self.clutch_dec = 0.01
        self.clutch_max_modifier = 1.3
        self.clutch_max_time = 1.5
        
        # For debugging
        self.debug_counter = 0
        
        # Constants for advanced cornering
        self.max_sensor_range = 200.0
        self.min_speed = 40.0
        self.max_normal_speed = 120.0
        
        # For turning estimation
        self.estimated_turn = 0.0
        self.previous_angles = []
        
        # For cornering
        self.min_agg_turn = -0.2
        self.max_agg_turn = 0.2
        self.min_agg_v = 0.5
        self.max_agg_v = 1.1
    
    def _toggle_driving_mode(self):
        """Callback for keyboard controller to toggle driving mode"""
        new_mode = self.MODE_MANUAL if self.mode == self.MODE_AI else self.MODE_AI
        self.set_mode(new_mode)
        print(f"DRIVING MODE CHANGED TO: {new_mode.upper()}")
    
    def _init_driver_params(self):
        """Initialize driver parameters based on track and car"""
        # Get track parameters if track is supported
        if self.track_name in self.track_params:
            track = self.track_params[self.track_name]
            self.max_speed = track["max_speed"]
            self.lookahead_dist = track["lookahead_dist"]
            self.steer_gain = track["steer_gain"]
            self.speed_gain = track["speed_gain"]
        else:
            # Default values
            self.max_speed = 100
            self.lookahead_dist = 10
            self.steer_gain = 1.0
            self.speed_gain = 1.0
            
        # Get car parameters if car is supported
        if self.car_name in self.car_params:
            car = self.car_params[self.car_name]
            
            # Basic parameters
            self.gear_up_rpm = car["gear_up_rpm"]
            self.gear_down_rpm = car["gear_down_rpm"]
            self.brake_power = car["brake_power"]
            self.accel_power = car["accel_power"]
            self.drivetrain = car["drivetrain"]
            self.steer_lock = car["steer_lock_rad"]
            self.brake_balance = car["brake_balance"]
            self.max_brake_pressure = car["max_brake_pressure"]
            self.gear_ratios = car["gear_ratios"]
            self.max_torque = car["max_torque"]
            self.max_torque_rpm = car["max_torque_rpm"]
            self.torque_curve = car["torque_curve"]
            self.abs_slip_threshold = car["abs_slip_threshold"]
            self.tc_slip_threshold = car["tc_slip_threshold"] 
            self.car_weight = car["weight"]
            self.tire_mu = car["tire_mu"]
            self.gear_efficiency = car["gear_efficiency"]
            self.central_diff = car["central_diff"]
            self.diff_type = car["diff_type"]
            self.diff_ratio = car["diff_ratio"]
            self.fuel_consumption = car["fuel_consumption"]
            self.suspension_travel = car["suspension_travel"]
            self.ground_clearance = car["ground_clearance"]
            self.car_width = car["car_width"]
            self.car_length = car["car_length"]
            
            # Set gear shift thresholds based on car's rev limiter and torque curve
            self.gear_up_thresholds = [0]
            for i in range(1, len(self.gear_ratios)):
                if i < len(self.gear_ratios) - 1:
                    # Shift at optimal RPM considering both power band and rev limit
                    optimal_upshift = min(self.gear_up_rpm, self.max_torque_rpm * 1.15)
                    self.gear_up_thresholds.append(optimal_upshift)
                else:
                    self.gear_up_thresholds.append(0)  # Last gear doesn't upshift
                    
            self.gear_down_thresholds = [0, 0]  # No downshift for neutral or 1st gear
            for i in range(2, len(self.gear_ratios)):
                # Calculate optimal downshift RPM based on torque curve
                torque_threshold = self.max_torque * 0.6
                downshift_rpm = self.gear_down_rpm
                
                # Find RPM where torque exceeds threshold (more sophisticated approach)
                for j in range(len(self.torque_curve) - 1):
                    rpm, torque = self.torque_curve[j]
                    if torque >= torque_threshold:
                        downshift_rpm = rpm
                        break
                
                self.gear_down_thresholds.append(downshift_rpm)
            
            # Car-specific cornering adjustments - MODIFIED for higher cornering speeds
            if self.drivetrain == "4WD":
                # 4WD can handle more aggressive cornering
                self.min_agg_v = 0.75  # Was 0.65, increased for higher speeds
                self.max_agg_v = 1.35  # Was 1.2, increased for higher speeds
            elif self.drivetrain == "FWD":
                # FWD needs more careful cornering
                self.min_agg_v = 0.6   # Was 0.5, increased for higher speeds
                self.max_agg_v = 1.15  # Was 1.0, increased for higher speeds
            else:  # RWD
                # RWD is balanced
                self.min_agg_v = 0.65  # Was 0.55, increased for higher speeds
                self.max_agg_v = 1.25  # Was 1.1, increased for higher speeds
                
            # Adjust cornering speeds based on car weight and grip
            weight_factor = 1350 / self.car_weight  # Normalized to ToyotaCorollaWRC (was 1300)
            grip_factor = self.tire_mu / 1.4        # More grip is better
            self.min_agg_v *= (weight_factor * grip_factor)
            self.max_agg_v *= (weight_factor * grip_factor)
            
            # Set a minimum speed global value based on car characteristics
            self.min_speed = 40.0  # Was 25.0, increased base minimum speed
            
            # Minimum speeds calculated from gear ratios and idle
            self.min_speeds = [0]
            for i in range(1, len(self.gear_ratios)):
                # Minimum viable speed in each gear to prevent lugging
                # Formula: wheel RPM = engine RPM / (gearRatio * diffRatio)
                # Speed = wheel RPM * wheel circumference / 60
                # Approximation based on typical wheel size
                min_engine_rpm = 1500  # Prevent lugging
                wheel_radius = 0.3     # meters (rough approximation)
                wheel_circumference = 2 * math.pi * wheel_radius
                
                # Calculate based on gear and diff ratios
                if self.gear_ratios[i] > 0:  # Skip reverse gear
                    wheel_rpm = min_engine_rpm / (self.gear_ratios[i] * self.diff_ratio)
                    min_speed_ms = wheel_rpm * wheel_circumference / 60
                    self.min_speeds.append(min_speed_ms)
                else:
                    self.min_speeds.append(0)
                    
        else:
            # Default values
            self.gear_up_rpm = 7000
            self.gear_down_rpm = 3000
            self.brake_power = 1.0
            self.accel_power = 1.0
            self.drivetrain = "4WD"
            self.steer_lock = 0.785398  # 45 degrees in radians
            self.brake_balance = 0.6
            self.max_brake_pressure = 15000
            self.gear_ratios = [-3.0, 3.0, 2.0, 1.5, 1.0, 0.8, 0.7]
            self.torque_curve = [(0, 0), (3000, 200), (5000, 300), (7000, 250), (8000, 200)]
            self.abs_slip_threshold = 3.0
            self.tc_slip_threshold = 5.0
            self.car_weight = 1300
            self.tire_mu = 1.5
            self.central_diff = None
            self.diff_type = "LIMITED SLIP"
            self.diff_ratio = 4.0
            self.suspension_travel = 0.4
            self.ground_clearance = 0.25
            self.car_width = 2.0
            self.car_length = 4.0
            
            # Default gear thresholds
            self.gear_up_thresholds = [0, 6500, 6500, 6500, 6500, 6500, 0]
            self.gear_down_thresholds = [0, 0, 2500, 3000, 3000, 3000, 3000]
            self.min_speeds = [0, 5, 15, 30, 50, 70, 90]
            
            # Default cornering parameters - MODIFIED for higher cornering speeds
            self.min_agg_v = 0.7   # Was 0.6, increased 
            self.max_agg_v = 1.3   # Was 1.1, increased
            self.min_speed = 40.0  # Was 25.0, increased
            
        # Track-specific settings for cornering adjusted by car parameters
        if self.track_name == "G-Speedway":  # Oval track - aggressive cornering
            self.min_agg_v *= 1.4  # Was 1.3, increased for higher speeds
            self.max_agg_v *= 1.4  # Was 1.3, increased for higher speeds
        elif self.track_name == "E-Track3":  # Road track - balanced cornering
            # Increase speed for road track as well
            self.min_agg_v *= 1.2  # Was 1.1, increased for higher speeds
            self.max_agg_v *= 1.2  # Was 1.1, increased for higher speeds
        elif self.track_name == "Dirt2":     # Dirt track - cautious cornering
            # Less reduction on dirt
            self.min_agg_v *= 0.9  # Was 0.85, less reduction for higher speeds
            self.max_agg_v *= 0.9  # Was 0.85, less reduction for higher speeds
            
            # On dirt, 4WD has a significant advantage
            if self.drivetrain == "4WD":
                self.min_agg_v *= 1.2  # Was 1.15, increased for higher speeds
                self.max_agg_v *= 1.2  # Was 1.15, increased for higher speeds
    
    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for x in range(19)]
        
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        
        return self.parser.stringify({'init': self.angles})
    
    def drive(self, msg):
        """Process sensor data and control the car"""
        self.state.setFromMsg(msg)
        
        if self.mode == self.MODE_MANUAL:
            # Manual driving mode - get controls from keyboard
            self._manual_drive()
        else:
            # AI driving mode
            self._ai_drive()
        
        # Limit the controls to valid ranges
        self._limit_controls()
        
        # Debug output (every 50 steps)
        self.debug_counter += 1
        if self.debug_counter % 50 == 0 and self.mode == self.MODE_AI:
            speed = self.state.getSpeedX()
            rpm = self.state.getRpm()
            gear = self.state.getGear()
            print(f"DEBUG: Speed={speed:.1f}, RPM={rpm:.0f}, Gear={gear}")
        
        return self.control.toMsg()
    
    def _limit_controls(self):
        """Apply limits to control values to ensure they're within valid ranges"""
        self.control.setAccel(max(0.0, min(1.0, self.control.getAccel())))
        self.control.setBrake(max(0.0, min(1.0, self.control.getBrake())))
        self.control.setSteer(max(-1.0, min(1.0, self.control.getSteer())))
        self.control.setClutch(max(0.0, min(1.0, self.control.getClutch())))
        self.control.setGear(max(-1, min(6, self.control.getGear())))
    
    def _manual_drive(self):
        """Get control inputs from keyboard controller"""
        controls = self.keyboard.get_controls()
        
        self.control.setAccel(controls['accel'])
        self.control.setBrake(controls['brake'])
        self.control.setSteer(controls['steer'])
        self.control.setGear(controls['gear'])
        self.control.setClutch(controls['clutch'])
    
    def _ai_drive(self):
        """AI driving logic based on sensor data"""
        # First, estimate the turn direction
        self._estimate_turn_direction()
        
        # Handle initial startup phase to ensure we go forward
        self._handle_startup()
        
        # Apply clutch control for smooth gear changes
        self._control_clutch()
        
        # Improved steering based on track sensors
        self._ai_steer()
        
        # Improved gear control
        self._ai_gear()
        
        # Improved speed control with ABS and traction control
        self._ai_speed()
    
    def _estimate_turn_direction(self):
        track = self.state.getTrack()
        if not track or len(track) < 5:
            self.estimated_turn = 0.0
            return
    
        mid = len(track) // 2
        angles_deg = self.angles
    
        # Increase focus range to look further ahead for turns
        # This prevents reacting too early to upcoming turns
        left_idx = max(0, mid - 3)  # Was mid-2, now looking wider
        right_idx = min(len(track) - 1, mid + 3)  # Was mid+2, now looking wider
    
        weighted_sum = 0.0
        total_weight = 0.0
    
        for i in range(left_idx, right_idx + 1):
            dist = track[i]
            angle_rad = math.radians(angles_deg[i])
            if dist > 0:
                # Prioritize more distant readings to detect turns earlier
                if abs(i - mid) <= 1:  # Center sensors
                    distance_weight = 0.8  # Reduce weight of immediate sensors
                else:
                    distance_weight = 1.0 + 0.2 * min(2, abs(i - mid))  # Increase weight of distant sensors
                    
                # Calculate weight - give more emphasis to forward readings
                weight = distance_weight / (1.0 + abs(angle_rad))
                lateral_offset = dist * math.sin(angle_rad)
                forward_dist = dist * math.cos(angle_rad)
                if forward_dist > 0:
                    curvature = lateral_offset / forward_dist
                    weighted_sum += curvature * weight
                    total_weight += weight
    
        if total_weight > 0:
            smoothed = weighted_sum / total_weight
            # Increase threshold to ignore minor turns - from 0.05 to 0.08
            self.estimated_turn = 0.0 if abs(smoothed) < 0.08 else smoothed
        else:
            self.estimated_turn = 0.0

    
    def _control_clutch(self):
        """Apply clutch control for smooth gear changes"""
        clutch = self.control.getClutch()
        
        # Apply clutch at race start
        if (self.state.getCurLapTime() < self.clutch_delta_time and 
                self.state.getDistRaced() < self.clutch_delta_raced):
            clutch = self.clutch_max
            
        # Adjust clutch value
        if clutch > 0:
            # Stronger clutch in lower gears or at start
            if self.state.getGear() < 2:
                if self.state.getCurLapTime() < self.clutch_max_time:
                    clutch = self.clutch_max * self.clutch_max_modifier
            
            # Decrease clutch gradually
            if clutch >= self.clutch_max:
                clutch -= self.clutch_dec  # Slow decrease at max
            else:
                clutch -= self.clutch_delta  # Faster decrease otherwise
                clutch = max(0.0, clutch)
        
        self.control.setClutch(clutch)
    
    def _handle_startup(self):
        """Handle initial startup to ensure the car moves forward properly"""
        speed = self.state.getSpeedX()
        
        # Initial phase - ensure car starts in first gear and moving forward
        if self.startup_counter < 100:  # First few seconds
            self.startup_counter += 1
            
            # Force first gear and full throttle to get moving
            if self.initial_accel_phase:
                self.control.setGear(1)
                self.control.setAccel(1.0)
                self.control.setBrake(0.0)
                self.control.setClutch(self.clutch_max)  # Apply clutch at start
                
                # If we're moving backward, apply brake to stop
                if speed < -0.5:
                    self.control.setAccel(0.0)
                    self.control.setBrake(1.0)
                # Once stopped, switch to forward
                elif abs(speed) < 0.5 and self.startup_counter > 20:
                    self.control.setAccel(1.0)
                    self.control.setBrake(0.0)
                    self.control.setGear(1)
                    self.control.setClutch(self.clutch_max)  # Apply clutch
                    if self.startup_counter > 30:
                        self.initial_accel_phase = False
    
    def _logistic_sigmoid(self, min_y, max_y, min_x, max_x, percent, x):
        """Implement logistic sigmoid function for smoother transitions"""
        if min_y > max_y:
            # Swap if min and max are reversed
            temp = max_y
            max_y = min_y
            min_y = temp
            temp = max_x
            max_x = min_x
            min_x = temp
            
        # Calculate constants for the logistic function based on given constraints
        c = math.log(((max_y - min_y/percent) * (percent*max_y - min_y)) / 
                    ((-min_y + min_y/percent) * (max_y - percent*max_y))) / (min_x - max_x)
        d = -max_x + math.log(((max_y - min_y) / (percent*max_y - min_y)) - 1.0) / c
        
        # Apply the logistic function
        a = max_y - min_y
        b = 1.0
        result = (a / (b + math.exp(c * (x + d)))) + min_y
        
        return result
    
    def _ai_steer(self):
        """Enhanced steering control using car-specific XML parameters"""
        track = self.state.getTrack()
        if not track:
            return
        
        angle = self.state.getAngle()
        dist_from_center = self.state.getTrackPos()
        speed = self.state.getSpeedX()
        
        # Basic steering behavior when off track
        if abs(dist_from_center) > 1.0:  # We're off track
            if self.state.getGear() == -1:  # If in reverse
                steer = -math.copysign(1.0, angle)  # Turn away from current angle
            else:
                # Recovery steering behavior based on drivetrain
                if self.drivetrain == "4WD":
                    # 4WD has better traction, can use more aggressive recovery
                    steer = math.copysign(1.0, angle) * 0.9
                elif self.drivetrain == "FWD":
                    # FWD tends to understeer offroad, need more steering input
                    steer = math.copysign(1.0, angle) * 1.0
                else:  # RWD
                    # RWD can oversteer, be more gentle
                    steer = math.copysign(1.0, angle) * 0.8
                
                # Apply additional steering based on speed (less at higher speeds)
                if speed > 50:
                    steer *= 0.7
                elif speed > 30:
                    steer *= 0.8
        else:
            # On-track steering - use a more sophisticated approach
            # Get the track sensor with maximum distance (usually the racing line)
            num_sensors = len(track)
            mid_sensor = num_sensors // 2
            
            # Find the best sensor (maximum distance)
            max_distance_sensor = mid_sensor
            max_distance = track[mid_sensor]
            
            for i in range(num_sensors):
                # Prefer sensors closer to middle for similar distances
                sensor_weight = 1.0 - 0.02 * abs(i - mid_sensor)
                weighted_dist = track[i] * sensor_weight
                if weighted_dist > max_distance:
                    max_distance = weighted_dist
                    max_distance_sensor = i
            
            # Calculate steering direction
            steer_sum = 0
            weight_sum = 0
            
            # MODIFIED: Look further ahead at higher speeds for smoother steering
            # Correction based on distance ahead and speed - more look-ahead at higher speeds
            correction_range = 3  # Default sensor range to consider
            if speed > 100:
                correction_range = 7  # Was 5, look even further ahead at high speed
            elif speed > 80:
                correction_range = 6
            elif speed > 50:
                correction_range = 5
            elif max_distance > 50:
                correction_range = 4
            
            # Drivetrain-specific steering adjustments
            soar_bias = 0.0  # Default no bias
            if self.drivetrain == "FWD":
                # FWD tends to understeer, bias more to inside of turn
                soar_bias = 0.15
                # MODIFIED: Reduce understeer compensation at high speed for smoother lines
                if speed > 80:
                    soar_bias *= 0.8
            elif self.drivetrain == "RWD":
                # RWD can oversteer, bias less to inside
                soar_bias = 0.05
            
            # MODIFIED: Adjust turn anticipation based on speed
            # Bias steering direction based on estimated turn and drivetrain
            # This helps the car prepare for upcoming turns
            soar_direction = 0.0
            if abs(self.estimated_turn) > 0.1:
                # Adjust turn anticipation factor based on speed - smoother at high speed
                soar_factor = 0.2
                if speed > 100:
                    soar_factor = 0.15  # Smoother, less aggressive at very high speed
                elif speed > 50:
                    soar_factor = 0.18  # Still smooth at medium-high speed
                
                # Amplify steering in the direction of the turn with drivetrain adjustment
                soar_direction = math.copysign(soar_factor + soar_bias, self.estimated_turn)
            
            # Apply car-specific racing line bias
            if self.car_name == "Peugeot406":
                # Road car - takes tighter racing line
                racing_line_bias = 0.1 if dist_from_center > 0 else -0.1
                # MODIFIED: Reduce turn-in at high speed for more stable cornering
                if speed > 80:
                    racing_line_bias *= 0.7
                soar_direction += racing_line_bias
            elif self.car_name == "ToyotaCorollaWRC":
                # Rally car - prefers wider exit from corner
                if abs(self.estimated_turn) > 0.15:
                    racing_line_bias = 0.05 if dist_from_center > 0 else -0.05
                    soar_direction += racing_line_bias
            
            # Calculate weighted steering direction with advanced sensor weighting
            for i in range(max_distance_sensor - correction_range, max_distance_sensor + correction_range + 1):
                if 0 <= i < num_sensors:
                    # Skip very small angles
                    if abs(self.angles[i]) < 2 and self.angles[i] != 0:
                        continue
                    
                    # Apply soar direction bias to make car move toward inside of turns
                    sensor_dist = track[i]
                    
                    # Weight sensors based on track curvature and car characteristics
                    if i < max_distance_sensor:  # Left side sensors
                        # Reduce weight on one side for turn-in
                        sensor_dist /= (1.0 + soar_direction)
                    elif i > max_distance_sensor:  # Right side sensors
                        # Increase weight on the other side
                        sensor_dist *= (1.0 + soar_direction)
                    
                    # Skip if angle too large (more than 90 degrees from current heading)
                    sensor_angle_rad = math.radians(self.angles[i])
                    if abs(angle + sensor_angle_rad) > math.pi/2:
                        continue
                    
                    # MODIFIED: Apply stronger distance-based weighting at higher speeds
                    # This helps smooth the steering at high speed by focusing more on distant points
                    distance_weight = 1.0
                    if abs(i - mid_sensor) > 0:
                        # High speed requires more distant focus for smooth steering
                        if speed > 80:
                            # Sensors further from center get more weight at high speed
                            # This smooths steering input by looking further ahead
                            center_weight = 0.05 * abs(i - mid_sensor)
                            distance_weight = 1.0 / (1.0 + 0.08 * abs(i - mid_sensor) - center_weight)
                        else:
                            # Normal weighting at lower speeds
                            distance_weight = 1.0 / (1.0 + 0.1 * abs(i - mid_sensor))
                    
                    # Add weighted contribution to steering
                    steer_sum += (self.angles[i] * sensor_dist * distance_weight)
                    weight_sum += (sensor_dist * distance_weight)
            
            # Calculate final steering value
            if weight_sum > 0:
                steer = -math.radians(steer_sum / weight_sum) / self.steer_lock
            else:
                # Fallback if no valid sensors
                steer = -angle / self.steer_lock
                
            # Apply drivetrain-specific track position correction
            if self.drivetrain == "FWD":
                # FWD needs more aggressive correction to counter understeer
                # MODIFIED: Adjust correction based on speed for smoother handling
                if speed > 80:
                    steer -= (dist_from_center * 0.5)  # Less aggressive at high speed
                else:
                    steer -= (dist_from_center * 0.6)
            elif self.drivetrain == "4WD":
                # 4WD has balanced handling
                steer -= (dist_from_center * 0.5)
            else:  # RWD
                # RWD needs less correction to avoid oversteer
                steer -= (dist_from_center * 0.4)
            
            # Apply car-specific width adjustment - wider cars need more caution near edges
            if abs(dist_from_center) > 0.7:  # Close to track edge
                width_factor = self.car_width / 2.0  # Normalize by average width
                steer -= math.copysign(0.2 * width_factor, dist_from_center)
        
        # Apply track-specific steering gain
        steer *= self.steer_gain
        
        # Adjust steering sensitivity based on car's actual steering lock from XML
        steer_sensitivity = 0.785398 / self.steer_lock  # Normalize to standard 45-degree
        steer *= steer_sensitivity
        
        # Reduce steering if moving backward to prevent spinning
        if speed < -1.0:
            steer *= 0.3
            
        # MODIFIED: Apply more progressive speed-based steering reduction
        # Less steering at high speeds for smoother, more stable handling
        if speed > 100:
            steer *= 0.7   # Was 0.8, reduced more for higher speed stability
        elif speed > 80:
            steer *= 0.8   # Same as before
        elif speed > 60:
            steer *= 0.85  # Same as before
        elif speed > 40:
            steer *= 0.9   # Same as before
        
        # MODIFIED: Apply steering damping at high speed to avoid oscillation
        if speed > 80 and abs(steer) < 0.1:
            # Apply additional damping to small steering inputs at high speed
            # This helps prevent oscillation and provides more stable handling
            steer *= 0.8
        
        # Limit steering when jumping to avoid spinning out
        if self.state.getZ() > 0.5:  # We're in the air
            steer *= 0.2
        
        # Drivetrain-specific high-speed adjustments
        if speed > 100:
            if self.drivetrain == "FWD":
                # FWD is more stable at high speed, can use more steering
                steer *= 0.9
            elif self.drivetrain == "RWD":
                # RWD needs more caution at high speed
                steer *= 0.8
        
        # Apply car weight and suspension characteristics to steering
        if self.car_name == "Peugeot406":
            # Heavier car with road suspension - slower response
            steer_damping = min(1.0, 1.0 - (0.001 * self.car_weight))
            steer *= steer_damping
        elif self.car_name == "ToyotaCorollaWRC" or self.car_name == "MitsubishiLancer":
            # Rally cars with rally suspension - more responsive
            steer *= 1.05
        
        self.control.setSteer(steer)
    
    def _ai_gear(self):
        """Enhanced gear control for AI using car-specific parameters from XML"""
        rpm = self.state.getRpm()
        gear = self.state.getGear()
        speed = self.state.getSpeedX()
        
        # Ensure we start in first gear, not reverse or neutral
        if gear <= 0:
            self.control.setGear(1)
            return
        
        # Get the number of gears for this car
        num_gears = len(self.gear_ratios) - 1  # Subtract 1 for reverse gear
        
        # Calculate current acceleration
        accel = self.control.getAccel()
        brake = self.control.getBrake()
        
        # Get current steering angle to determine if in a turn
        steer = abs(self.control.getSteer())
        in_turn = steer > 0.1
        
        # Get optimal RPM from torque curve - we want to stay near max torque for best acceleration
        rpm_at_max_torque = self.max_torque_rpm
        
        # Adjust shift points based on drivetrain type and driving conditions
        if self.drivetrain == "FWD":
            # FWD cars need higher RPM in turns for better handling
            if in_turn and accel > 0.5:
                # Prevent upshift in turns for FWD cars to maintain power through corners
                upshift_threshold = self.gear_up_thresholds[gear] * 1.1
            else:
                upshift_threshold = self.gear_up_thresholds[gear]
                
            # FWD cars benefit from earlier downshifts when braking
            if brake > 0.2:
                downshift_threshold = self.gear_down_thresholds[gear] * 1.2
            else:
                downshift_threshold = self.gear_down_thresholds[gear]
        
        elif self.drivetrain == "4WD":
            # 4WD cars have better traction, so we can shift earlier to save fuel
            if in_turn:
                # Need less RPM in turns due to better traction
                upshift_threshold = self.gear_up_thresholds[gear] * 0.95
            else:
                upshift_threshold = self.gear_up_thresholds[gear]
                
            # 4WD can use engine braking more effectively
            if brake > 0.3:
                downshift_threshold = self.gear_down_thresholds[gear] * 1.1
            else:
                downshift_threshold = self.gear_down_thresholds[gear]
        
        else:  # RWD
            # RWD cars need careful power control
            if in_turn and accel > 0.7:
                # Prevent upshift in aggressive turns to avoid oversteer
                upshift_threshold = self.gear_up_thresholds[gear] * 1.05
            else:
                upshift_threshold = self.gear_up_thresholds[gear]
                
            downshift_threshold = self.gear_down_thresholds[gear]
        
        # Check for minimum speed in current gear - prevent lugging the engine
        if gear > 1 and speed < self.min_speeds[gear] and rpm < self.gear_down_rpm:
            self.control.setGear(gear - 1)
            return
            
        # Check if we're at redline and need to upshift
        if gear < num_gears and rpm >= upshift_threshold:
            # Don't upshift if we're in a turn and need the power (depending on car)
            if not (in_turn and accel > 0.8 and self.drivetrain != "4WD"):
                self.control.setGear(gear + 1)
            return
            
        # Check if RPM is too low and we need to downshift
        if gear > 1 and rpm <= downshift_threshold:
            # For acceleration, downshift to get closer to optimal power band
            if accel > 0.5 and rpm < rpm_at_max_torque * 0.7:
                self.control.setGear(gear - 1)
                return
                
            # For braking/deceleration, downshift for engine braking
            if brake > 0.2:
                self.control.setGear(gear - 1)
                return
                
        # Special case for specific cars with FWD - more aggressive downshifting in turns
        if self.drivetrain == "FWD" and in_turn:
            # For FWD cars, lower gear helps with turn-in and reduces understeer
            if gear > 1 and rpm < rpm_at_max_torque * 0.8:
                self.control.setGear(gear - 1)
                return
        
        # Keep current gear otherwise
        self.control.setGear(gear)
    
    def _ai_speed(self):
        """Enhanced speed control using car-specific XML parameters"""
        speed = self.state.getSpeedX()
        track = self.state.getTrack()
        
        if not track:
            return
        
        # Get the steering value for the current state
        current_steer = self.control.getSteer()
        
        # Get the distance straight ahead (center sensor)
        mid = len(track) // 2
        dist_front = track[mid]
        
        # Different behavior if we're stuck or off-track
        if abs(self.state.getTrackPos()) > 1.0:  # We're off track
            if self.state.getGear() < 0:  # In reverse
                target_speed = self.min_speed
                accel = 0.5  # Gentle acceleration
            else:
                # Off-track recovery behavior varies by drivetrain
                if self.drivetrain == "4WD":
                    # 4WD has better off-road capability
                    if abs(current_steer) < 0.2:  # We're pointing in a decent direction
                        # Push harder to get back on track - 4WD has best traction
                        target_speed = max(self.min_speed * 4.5, speed)
                        accel = 0.9
                    else:  # We're still turning to get back on track
                        target_speed = self.min_speed * 3.5
                        accel = 0.4
                elif self.drivetrain == "FWD":
                    # FWD is more prone to understeer offroad
                    if abs(current_steer) < 0.1:  # Need to be pointing more precisely
                        target_speed = max(self.min_speed * 3.5, speed)
                        accel = 0.7
                    else:
                        # FWD needs to be more careful off-track
                        target_speed = self.min_speed * 2.5
                        accel = 0.3
                else:  # RWD
                    # RWD is most likely to spin off-track
                    if abs(current_steer) < 0.15:
                        target_speed = max(self.min_speed * 4.0, speed)
                        accel = 0.75
                    else:
                        target_speed = self.min_speed * 2.0
                        accel = 0.25
            
            brake = 0.0
        else:
            # On track - calculate target speed based on turn conditions
            if self.state.getGear() < 0:
                target_speed = self.min_speed
            else:
                # Calculate target speed based on distance and turn angle
                min_speed = self.min_speed
                max_speed = self.max_normal_speed * self.speed_gain
                
                # Apply car-specific speed adjustments
                if self.car_name == "ToyotaCorollaWRC":
                    # Rally car - good on varied surfaces
                    if self.track_name == "Dirt2":
                        max_speed *= 1.1  # Better on dirt
                    elif self.track_name == "G-Speedway":
                        max_speed *= 0.95  # Not optimal for pure speed tracks
                elif self.car_name == "Peugeot406":
                    # Road car - good on tarmac
                    if self.track_name == "Dirt2":
                        max_speed *= 0.85  # Worse on dirt
                    elif self.track_name == "G-Speedway":
                        max_speed *= 1.05  # Better on smooth tracks
                elif self.car_name == "MitsubishiLancer":
                    # Evo - balanced performance
                    if self.track_name == "Dirt2":
                        max_speed *= 1.1  # Good on dirt
                    elif self.track_name == "G-Speedway":
                        max_speed *= 1.0  # Good on all surfaces
                
                # Calculate weight-to-power ratio effect
                power_to_weight = 1.0  # Default
                if self.car_name == "ToyotaCorollaWRC":
                    power_to_weight = 1.05
                elif self.car_name == "Peugeot406":
                    power_to_weight = 0.95
                elif self.car_name == "MitsubishiLancer":
                    power_to_weight = 1.1
                
                # Adjust max speed based on power-to-weight ratio
                max_speed *= power_to_weight
                
                # MODIFIED: Make turn aggression more aggressive (less slowing down)
                # Increase the aggression values to maintain more speed in corners
                turn_aggression_boost = 0.5  # Higher value = less slowing down in corners (was 0.4)
                
                # Calculate turn aggression with boosted values
                turn_aggression = self._logistic_sigmoid(
                    self.min_agg_v, self.max_agg_v + turn_aggression_boost,
                    self.min_agg_turn, self.max_agg_turn,
                    0.995, abs(self.estimated_turn)
                )
                
                # Adjust by drivetrain type and surface
                if self.drivetrain == "FWD":
                    # FWD: More cautious in corners, better traction under acceleration
                    if abs(self.estimated_turn) > 0.2:  # In a significant turn (was 0.15)
                        # MODIFIED: Reduce understeer penalty
                        turn_aggression *= 0.92  # Was 0.9, less reduction for higher speeds
                    else:
                        turn_aggression *= 0.97  # Was 0.95, less reduction
                elif self.drivetrain == "4WD":
                    # 4WD: Best cornering, good traction - increase aggression
                    if self.track_name == "Dirt2":
                        turn_aggression *= 1.25  # Even better on dirt (was 1.2)
                    else:
                        turn_aggression *= 1.2  # Generally more aggressive (was 1.15)
                else:  # RWD
                    # RWD: Balanced, but can oversteer
                    if abs(self.estimated_turn) > 0.2:  # In a significant turn (was 0.15)
                        turn_aggression *= 0.97  # Was 0.95, less reduction
                    else:
                        turn_aggression *= 1.15  # Was 1.1, more aggressive on straights
                
                # Calculate target speed using the aggression exponent and distance
                # MODIFIED: Square root the distance factor for less speed reduction at medium distances
                dist_factor = math.pow(dist_front / self.max_sensor_range, turn_aggression * 0.75)  # Was 0.85
                target_speed = (dist_factor * (max_speed - min_speed)) + min_speed
                
                # MODIFIED: Increase minimum corner speed based on car type
                # Only apply minimum corner speed for significant turns (increased threshold)
                if abs(self.estimated_turn) > 0.15:  # Was 0.1
                    # Set a higher minimum cornering speed
                    min_corner_speed = 50.0  # Was calculated as min_speed * 2.0 (typically ~37.5)
                    
                    # Adjust minimum corner speed by car type
                    if self.drivetrain == "4WD":
                        min_corner_speed *= 1.4  # Rally cars can corner faster (was 1.3)
                    elif self.drivetrain == "FWD":
                        min_corner_speed *= 1.2  # Road cars corner slower (was 1.1)
                    
                    # Apply speed scaling based on how sharp the turn is
                    turn_sharpness = min(1.0, abs(self.estimated_turn) / 0.3)
                    # Reduce min speed for very sharp turns, but maintain higher minimum
                    min_corner_speed *= max(0.8, 1.0 - turn_sharpness * 0.2)
                        
                    # Ensure we don't go below our minimum corner speed
                    target_speed = max(target_speed, min_corner_speed)
                
                # Apply speed limits based on track position and steering, but less aggressively
                if abs(self.state.getTrackPos()) > 0.85:  # Near edge (was 0.8)
                    target_speed *= 0.9  # Be somewhat cautious near track edges (was 0.85)
                
                # Apply minimum speed constraint
                target_speed = max(target_speed, min_speed)
                
                # Apply maximum speed constraint
                target_speed = min(target_speed, max_speed)
            
            # Calculate acceleration using sigmoid function - smoother control
            accel_coef = 1.5  # Steeper sigmoid for more responsive control
            
            # Calculate sigmoid input
            sigmoid_input = speed - target_speed
            
            # Use sigmoid function to calculate acceleration/brake
            output = 2.0 / (1.0 + math.exp(accel_coef * sigmoid_input)) - 1.0
            
            # Split into acceleration and braking
            if output >= 0:
                accel = output
                brake = 0.0
            else:
                accel = 0.0
                # MODIFIED: Reduce brake intensity to maintain more speed in corners
                brake = -output * self.brake_balance * 0.8  # Apply brake balance with reduction (was 0.9)
                
                # Apply more brake pressure in higher gears - better engine braking
                gear_factor = min(1.0, self.state.getGear() / 3.0)
                brake *= (1.0 + 0.15 * gear_factor)  # Was 0.2, reduced for less braking
        
        # Apply car-specific power and brake adjustments
        accel *= self.accel_power
        brake *= self.brake_power
        
        # Implement enhanced ABS (Anti-lock Braking System)
        if brake > 0.0:
            wheel_spin = self.state.getWheelSpinVel()
            if wheel_spin:
                # Check for wheel locking using car-specific threshold
                for i in range(min(4, len(wheel_spin))):
                    wheel_slip = abs(speed - wheel_spin[i])
                    if wheel_spin[i] < self.abs_slip_threshold and speed > 10.0:
                        # Wheel might be locking, apply ABS based on drivetrain
                        if i < 2:  # Front wheels
                            # Front wheel ABS is most important 
                            front_brake_factor = 0.65 if self.drivetrain == "FWD" else 0.7
                            brake *= front_brake_factor
                        else:  # Rear wheels
                            # Rear wheel ABS depends on drivetrain
                            if self.drivetrain == "4WD":
                                brake *= 0.7
                            elif self.drivetrain == "FWD":
                                brake *= 0.8  # Less critical for FWD
                            else:  # RWD
                                brake *= 0.65  # More critical for RWD
                        break
        
        # Implement enhanced traction control
        if accel > 0.0:
            wheel_spin = self.state.getWheelSpinVel()
            if wheel_spin and len(wheel_spin) >= 4:
                # Check for wheelspin based on drivetrain type
                drive_slip = 0
                
                if self.drivetrain == "4WD":
                    # 4WD: check all wheels
                    front_slip = max(wheel_spin[0], wheel_spin[1]) - speed
                    rear_slip = max(wheel_spin[2], wheel_spin[3]) - speed
                    drive_slip = max(front_slip, rear_slip)
                elif self.drivetrain == "FWD":
                    # FWD: check front wheels (0,1)
                    drive_slip = max(wheel_spin[0], wheel_spin[1]) - speed
                else:
                    # RWD: check rear wheels (2,3)
                    drive_slip = max(wheel_spin[2], wheel_spin[3]) - speed
                
                if drive_slip > self.tc_slip_threshold:
                    # Reduce acceleration to prevent wheelspin
                    tc_factor = 0
                    if self.drivetrain == "4WD":
                        tc_factor = 0.8  # 4WD has best traction
                    elif self.drivetrain == "FWD":
                        tc_factor = 0.7  # FWD less traction in acceleration
                    else:  # RWD
                        tc_factor = 0.65  # RWD most prone to wheelspin
                    
                    accel *= tc_factor
                    
                    # Progressive traction control
                    if drive_slip > self.tc_slip_threshold * 1.5:
                        accel *= 0.8  # Further reduce for excessive slip
        
        # Final safety checks - prevent sudden changes
        if self.state.getSpeedX() > 50 and abs(current_steer) > 0.4:
            # At high speed in tight corners, ensure we don't accelerate too much
            accel = min(accel, 0.5)
        
        self.control.setAccel(accel)
        self.control.setBrake(brake)
    
    def set_mode(self, mode):
        """Change driving mode between AI and manual"""
        if mode in [self.MODE_AI, self.MODE_MANUAL]:
            prev_mode = self.mode
            self.mode = mode
            
            # Reset startup sequence if switching to AI mode
            if mode == self.MODE_AI:
                self.startup_counter = 0
                self.initial_accel_phase = True
            
            # Start/stop keyboard controller as needed
            if mode == self.MODE_MANUAL and prev_mode != self.MODE_MANUAL:
                self.keyboard.start()
            elif mode == self.MODE_AI and prev_mode == self.MODE_MANUAL:
                pass  # Keep keyboard running for mode toggle
                
            print(f"Driving mode changed to: {mode}")
            return True
        return False
    
    def get_mode(self):
        """Get current driving mode"""
        return self.mode
    
    def set_track(self, track_name):
        """Set track name and update parameters"""
        self.track_name = track_name
        self._init_driver_params()
        
    def set_car(self, car_name):
        """Set car name and update parameters"""
        self.car_name = car_name
        self._init_driver_params()
    
    def onShutDown(self):
        """Called when the server shuts down"""
        if self.keyboard.running:
            self.keyboard.stop()
    
    def onRestart(self):
        """Called when the race is restarted"""
        # Reset control and state
        self.control = carControl.CarControl()
        self.prev_rpm = None
        
        # Reset keyboard controller if in manual mode
        if self.keyboard.running:
            self.keyboard.reset_controls()
            
        # Reset startup sequence
        self.startup_counter = 0
        self.initial_accel_phase = True
        