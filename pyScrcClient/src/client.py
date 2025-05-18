import sys
import argparse
import socket
import time
import driver
import logger

def main():
    # Configure the argument parser
    parser = argparse.ArgumentParser(description='Python client for TORCS with AI and manual driving support')
    
    parser.add_argument('--host', action='store', dest='host_ip', default='localhost',
                        help='Host IP address (default: localhost)')
    parser.add_argument('--port', action='store', type=int, dest='host_port', default=3001,
                        help='Host port number (default: 3001)')
    parser.add_argument('--id', action='store', dest='id', default='SCR',
                        help='Bot ID (default: SCR)')
    parser.add_argument('--maxEpisodes', action='store', dest='max_episodes', type=int, default=1,
                        help='Maximum number of learning episodes (default: 1)')
    parser.add_argument('--maxSteps', action='store', dest='max_steps', type=int, default=0,
                        help='Maximum number of steps (default: 0)')
    parser.add_argument('--track', action='store', dest='track', default='unknown',
                        help='Name of the track (G-Speedway, E-Track3, Dirt2)')
    parser.add_argument('--car', action='store', dest='car', default='ToyotaCorollaWRC',
                        help='Name of the car (ToyotaCorollaWRC, Peugeot406, MitsubishiLancer)')
    parser.add_argument('--stage', action='store', dest='stage', type=int, default=3,
                        help='Stage (0 - Warm-Up, 1 - Qualifying, 2 - Race, 3 - Unknown)')
    parser.add_argument('--mode', action='store', dest='mode', default='ai',
                        help='Driving mode (ai or manual)')
    parser.add_argument('--run_id', action='store', dest='run_id', type=int, default=1,
                        help='Run ID for CSV logging (default: 1)')
                        
    arguments = parser.parse_args()
    
    # Validate arguments
    if arguments.mode not in ['ai', 'manual']:
        print("Error: Mode must be 'ai' or 'manual'")
        sys.exit(-1)
        
    # Validate car argument
    valid_cars = ["ToyotaCorollaWRC", "Peugeot406", "MitsubishiLancer"]
    if arguments.car not in valid_cars:
        print(f"Warning: Car '{arguments.car}' is not in the supported list: {', '.join(valid_cars)}")
        print(f"Defaulting to ToyotaCorollaWRC")
        arguments.car = "ToyotaCorollaWRC"
        
    # Validate track argument
    valid_tracks = ["G-Speedway", "E-Track3", "Dirt2"]
    if arguments.track not in valid_tracks:
        print(f"Warning: Track '{arguments.track}' is not in the supported list: {', '.join(valid_tracks)}")
        print(f"Some optimizations may not be applied")
        
    # Print summary
    print('TORCS Python Client')
    print('--------------------------')
    print(f'Connecting to server host ip: {arguments.host_ip}, port: {arguments.host_port}')
    print(f'Bot ID: {arguments.id}')
    print(f'Maximum episodes: {arguments.max_episodes}')
    print(f'Maximum steps: {arguments.max_steps}')
    print(f'Track: {arguments.track}')
    print(f'Car: {arguments.car}')
    
    # Display car information
    car_info = {
        "ToyotaCorollaWRC": "4WD Rally Car - Balanced handling with good off-road capability",
        "Peugeot406": "FWD Road Car - Good on tarmac, tends to understeer",
        "MitsubishiLancer": "4WD Rally Car - Powerful with excellent traction"
    }
    
    track_info = {
        "G-Speedway": "Oval track - High speed, gentle corners",
        "E-Track3": "Road track - Mix of slow and fast corners",
        "Dirt2": "Dirt track - Loose surface, challenging traction"
    }
    
    if arguments.car in car_info:
        print(f'Car Info: {car_info[arguments.car]}')
    
    if arguments.track in track_info:
        print(f'Track Info: {track_info[arguments.track]}')
        
    print(f'Stage: {arguments.stage}')
    print(f'Mode: {arguments.mode}')
    print(f'Run ID: {arguments.run_id}')
    print('--------------------------')
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except socket.error as msg:
        print('Could not make a socket:', msg)
        sys.exit(-1)
    
    # One second timeout
    sock.settimeout(1.0)
    
    # Create driver and logger
    driver_instance = driver.Driver(
        stage=arguments.stage,
        mode=arguments.mode,
        car_name=arguments.car,
        track_name=arguments.track
    )
    
    # Create logger
    data_logger = logger.Logger()
    
    # Initialize logging
    if arguments.track in driver_instance.SUPPORTED_TRACKS and arguments.car in driver_instance.SUPPORTED_CARS:
        log_file = data_logger.initialize_log(
            track_name=arguments.track,
            car_name=arguments.car,
            mode=arguments.mode,
            run_id=arguments.run_id
        )
        print(f"Logging to file: {log_file}")
    else:
        print("Warning: Unsupported track or car name for logging. Using defaults.")
        log_file = data_logger.initialize_log(
            track_name=arguments.track,
            car_name=arguments.car,
            mode=arguments.mode,
            run_id=arguments.run_id
        )
    
    # Print car setup information
    if arguments.car in driver_instance.car_params:
        car = driver_instance.car_params[arguments.car]
        print("\nCar Setup Information:")
        print(f"Drivetrain: {car['drivetrain']}")
        print(f"Gear Up RPM: {car['gear_up_rpm']}")
        print(f"Max Torque: {car['max_torque']} Nm @ {car['max_torque_rpm']} RPM")
        print(f"Weight: {car['weight']} kg")
        if 'gear_ratios' in car:
            ratios = car['gear_ratios']
            print(f"Gears: {len(ratios)-1} forward, 1 reverse")
            if len(ratios) > 1:
                print(f"1st Gear Ratio: {ratios[1]:.3f}")
            if len(ratios) > 2:
                print(f"Top Gear Ratio: {ratios[-1]:.3f}")
    
    # Mode switch command
    print("\nControls:")
    print("Use keyboard to control the car in manual mode or toggle modes with 'M' key")
    print("W/S: Accelerate/Brake | A/D: Steer | Q/E: Shift down/up | Space: Clutch")
    print("Make sure the TORCS window is in focus for keyboard controls to work properly")
    
    # Main loop
    shutdownClient = False
    curEpisode = 0
    
    while not shutdownClient:
        # Identification loop
        while True:
            print('Sending id to server:', arguments.id)
            buf = arguments.id + driver_instance.init()
            
            try:
                sock.sendto(buf.encode('utf-8'), (arguments.host_ip, arguments.host_port))
            except socket.error as msg:
                print("Failed to send data...Exiting...")
                sys.exit(-1)
                
            try:
                buf, addr = sock.recvfrom(1000)
                buf = buf.decode('utf-8')
            except socket.error as msg:
                print("didn't get response from server...")
                continue
        
            if buf.find('***identified***') >= 0:
                print('Received:', buf)
                break
    
        # Simulation loop
        currentStep = 0
        
        # If we start in manual mode, make sure to initialize keyboard controller properly
        if driver_instance.get_mode() == driver_instance.MODE_MANUAL:
            print("Manual driving mode active - keyboard controls enabled")
            if not driver_instance.keyboard.running:
                driver_instance.keyboard.start()
        
        while True:
            # Mode toggle is handled directly through the keyboard controller now
            
            # Wait for server response
            buf = None
            try:
                buf, addr = sock.recvfrom(1000)
                buf = buf.decode('utf-8')
            except socket.error as msg:
                print("didn't get response from server...")
            
            # Handle server messages
            if buf is not None:
                if buf.find('***shutdown***') >= 0:
                    driver_instance.onShutDown()
                    data_logger.close()
                    shutdownClient = True
                    print('Client Shutdown')
                    break
                
                if buf.find('***restart***') >= 0:
                    driver_instance.onRestart()
                    data_logger.close()
                    print('Client Restart')
                    break
                
                # Process this simulation step
                currentStep += 1
                if currentStep != arguments.max_steps:
                    response = driver_instance.drive(buf)
                    
                    # Log data
                    data_logger.log_data(driver_instance.state, driver_instance.control)
                    
                    # Debug output - helpful to see state
                    if currentStep % 100 == 0:
                        mode = driver_instance.get_mode()
                        speed = driver_instance.state.getSpeedX()
                        rpm = driver_instance.state.getRpm()
                        gear = driver_instance.state.getGear()
                        dist = driver_instance.state.getDistRaced()
                        
                        print(f"Step {currentStep}: Mode={mode}, Speed={speed:.1f}m/s, "
                              f"RPM={rpm:.0f}, Gear={gear}, Distance={dist:.1f}m")
                    
                    # Detailed debug for manual mode
                    if driver_instance.get_mode() == driver_instance.MODE_MANUAL and currentStep % 20 == 0:
                        controls = driver_instance.keyboard.get_controls()
                        print(f"Manual Controls: Speed={driver_instance.state.getSpeedX():.1f} "
                              f"Accel={controls['accel']:.1f} Brake={controls['brake']:.1f} "
                              f"Steer={controls['steer']:.1f} Gear={controls['gear']}")
                else:
                    response = '(meta 1)'
                
                # Send response to server
                try:
                    sock.sendto(response.encode('utf-8'), (arguments.host_ip, arguments.host_port))
                except socket.error as msg:
                    print("Failed to send data...Exiting...")
                    sys.exit(-1)
        
        # Increment episode counter
        curEpisode += 1
        
        # Check if we've completed all episodes
        if curEpisode == arguments.max_episodes:
            shutdownClient = True
    
    # Clean up
    sock.close()
    print("Client finished successfully.")

if __name__ == "__main__":
    main() 