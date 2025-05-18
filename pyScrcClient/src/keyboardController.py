import threading
import time
from pynput import keyboard

class KeyboardController:
    """
    Class to handle keyboard inputs for manual driving in TORCS
    Uses a separate thread to continuously monitor keyboard state
    """
    
    def __init__(self):
        """Constructor"""
        self.running = False
        self.keyboard_thread = None
        self.listener = None
        
        # Control state
        self.accel = 0.0
        self.brake = 0.0
        self.steer = 0.0
        self.gear = 1
        self.clutch = 0.0
        
        # Currently pressed keys
        self.keys_pressed = set()
        self.last_gear_time = 0
        self.last_mode_toggle_time = 0
        
        # Mode toggle callback
        self.mode_toggle_callback = None
        
    def start(self):
        """Start the keyboard monitoring thread"""
        if not self.running:
            self.running = True
            self.keyboard_thread = threading.Thread(target=self._keyboard_monitor)
            self.keyboard_thread.daemon = True
            self.keyboard_thread.start()
            
            # Setup keyboard listener
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release)
            self.listener.start()
            
            print("Keyboard controller started. Controls:")
            print("  W/S: Accelerate/Brake")
            print("  A/D: Steer left/right")
            print("  Q/E: Shift down/up")
            print("  Space: Clutch")
            print("  R: Reset controls")
            print("  M: Toggle AI/manual mode")
            
    def stop(self):
        """Stop the keyboard monitoring thread"""
        self.running = False
        if self.listener:
            self.listener.stop()
            self.listener = None
        if self.keyboard_thread:
            self.keyboard_thread.join(timeout=1.0)
            self.keyboard_thread = None
            
    def set_mode_toggle_callback(self, callback):
        """Set a callback function to be called when mode is toggled with 'M' key"""
        self.mode_toggle_callback = callback
            
    def _on_press(self, key):
        """Called when a key is pressed"""
        try:
            # Convert key to character if possible
            if hasattr(key, 'char'):
                key_char = key.char.lower()
                self.keys_pressed.add(key_char)
            elif key == keyboard.Key.space:
                self.keys_pressed.add('space')
        except AttributeError:
            pass
            
    def _on_release(self, key):
        """Called when a key is released"""
        try:
            if hasattr(key, 'char'):
                key_char = key.char.lower()
                if key_char in self.keys_pressed:
                    self.keys_pressed.remove(key_char)
            elif key == keyboard.Key.space:
                if 'space' in self.keys_pressed:
                    self.keys_pressed.remove('space')
        except AttributeError:
            pass
            
    def _keyboard_monitor(self):
        """Thread function to monitor keyboard state and apply controls"""
        while self.running:
            try:
                # Process currently pressed keys
                self._process_keys()
                
                # Apply control decay (for smooth control)
                self._apply_control_decay()
                
                # Small sleep to reduce CPU usage
                time.sleep(0.05)
            except Exception as e:
                print(f"Keyboard monitoring error: {e}")
                
    def _process_keys(self):
        """Process currently pressed keys"""
        current_time = time.time()
        
        # Mode toggle (with time debounce)
        if 'm' in self.keys_pressed and current_time - self.last_mode_toggle_time > 0.5:
            if self.mode_toggle_callback:
                self.mode_toggle_callback()
            self.last_mode_toggle_time = current_time
            # Remove to prevent repeated toggles
            self.keys_pressed.discard('m')
        
        # Steering
        if 'a' in self.keys_pressed:
            self.steer = max(-1.0, self.steer + 0.05)
        if 'd' in self.keys_pressed:
            self.steer = min(1.0, self.steer - 0.05)
            
        # Acceleration/Braking
        if 'w' in self.keys_pressed:
            self.accel = min(1.0, self.accel + 0.05)
            self.brake = 0.0
        if 's' in self.keys_pressed:
            self.brake = min(1.0, self.brake + 0.05)
            self.accel = 0.0
            
        # Gear shifting (with time debounce)
        if 'e' in self.keys_pressed and current_time - self.last_gear_time > 0.3:
            self.gear = min(6, self.gear + 1)
            self.last_gear_time = current_time
            # Remove to prevent repeated gear shifts
            self.keys_pressed.discard('e')
            
        if 'q' in self.keys_pressed and current_time - self.last_gear_time > 0.3:
            self.gear = max(-1, self.gear - 1)
            self.last_gear_time = current_time
            # Remove to prevent repeated gear shifts
            self.keys_pressed.discard('q')
            
        # Clutch
        if 'space' in self.keys_pressed:
            self.clutch = 1.0
            
        # Reset controls
        if 'r' in self.keys_pressed:
            self.reset_controls()
            # Remove to prevent repeated resets
            self.keys_pressed.discard('r')
            
    def _apply_control_decay(self):
        """Apply decay to controls for smoother driving experience"""
        # Steering decay when not actively steering
        if 'a' not in self.keys_pressed and 'd' not in self.keys_pressed:
            if abs(self.steer) < 0.05:
                self.steer = 0.0
            elif self.steer > 0:
                self.steer = max(0, self.steer - 0.02)
            elif self.steer < 0:
                self.steer = min(0, self.steer + 0.02)
            
        # Acceleration decay when not pressing
        if 'w' not in self.keys_pressed and self.accel > 0:
            self.accel = max(0, self.accel - 0.02)
            
        # Brake decay when not pressing
        if 's' not in self.keys_pressed and self.brake > 0:
            self.brake = max(0, self.brake - 0.02)
            
        # Clutch decay
        if 'space' not in self.keys_pressed:
            self.clutch = max(0.0, self.clutch - 0.05)
            
    def reset_controls(self):
        """Reset all controls to default values"""
        self.accel = 0.0
        self.brake = 0.0
        self.steer = 0.0
        self.clutch = 0.0
        
    def get_controls(self):
        """Get the current control values"""
        return {
            'accel': self.accel,
            'brake': self.brake,
            'steer': self.steer,
            'gear': self.gear,
            'clutch': self.clutch
        } 