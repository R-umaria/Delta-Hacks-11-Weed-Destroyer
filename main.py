import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
import RPi.GPIO as GPIO
import time
import threading

# ---------------------- GPIO Setup ----------------------
GPIO.setwarnings(False)  # Suppress warnings about reused GPIO pins
GPIO.setmode(GPIO.BCM)

# Pin configuration for motors
motor_pins = {
    'left1': {'dir1': 5, 'dir2': 6, 'pwm': 13},  # Left motor 1
    'left2': {'dir1': 17, 'dir2': 27, 'pwm': 22},  # Left motor 2
    'right1': {'dir1': 11, 'dir2': 9, 'pwm': 10},  # Right motor 1
    'right2': {'dir1': 4, 'dir2': 3, 'pwm': 2},  # Right motor 2
}

# Pin configuration for servos
servo_pins = {"servo1": 19}
servo_pwms = {}

# Initialize motor pins
for motor in motor_pins.values():
    GPIO.setup(motor['dir1'], GPIO.OUT)
    GPIO.setup(motor['dir2'], GPIO.OUT)
    GPIO.setup(motor['pwm'], GPIO.OUT)
    motor['pwm_instance'] = GPIO.PWM(motor['pwm'], 1000)  # Set PWM frequency to 1kHz
    motor['pwm_instance'].start(0)  # Start with motor off

# Initialize servos
for servo, pin in servo_pins.items():
    GPIO.setup(pin, GPIO.OUT)
    pwm = GPIO.PWM(pin, 50)  # Servo operates at 50Hz
    pwm.start(7.5)  # Center position (7.5% duty cycle)
    servo_pwms[servo] = pwm

# ---------------------- Helper Functions ----------------------
def move(direction, speed = 15):
    """
    Controls the movement of the rover.
    :param direction: "forward" or "stop".
    :param speed: Speed percentage (0-100).
    """
    speed = max(0, min(100, speed))  # Ensure speed is within bounds
    if direction == "forward":
        for motor in motor_pins.values():
            GPIO.output(motor['dir1'], GPIO.HIGH)
            GPIO.output(motor['dir2'], GPIO.LOW)
            motor['pwm_instance'].ChangeDutyCycle(speed)
    elif direction == "stop":
        for motor in motor_pins.values():
            GPIO.output(motor['dir1'], GPIO.LOW)
            GPIO.output(motor['dir2'], GPIO.LOW)
            motor['pwm_instance'].ChangeDutyCycle(0)

def angle_to_duty_cycle(angle):
    """
    Converts an angle (0 to 180) to the corresponding PWM duty cycle for the servo.
    :param angle: Desired angle (0 to 180 degrees)
    :return: Duty cycle value
    """
    return 2.5 + (angle / 18.0)  # Map 0-180 degrees to 2.5%-12.5% duty cycle

def move_servo_smooth(servo_name, start_angle, target_angle, duration=2.0, steps=50):
    """
    Smoothly moves a servo from the start_angle to the target_angle using interpolation.
    :param servo_name: Name of the servo ("servo1")
    :param start_angle: Starting angle (0 to 180 degrees)
    :param target_angle: Target angle (0 to 180 degrees)
    :param duration: Total time (seconds) for the full movement
    :param steps: Number of steps for interpolation
    """
    if servo_name not in servo_pwms:
        print(f"Servo {servo_name} not found!")
        return

    pwm = servo_pwms[servo_name]
    angles = np.linspace(start_angle, target_angle, steps)
    delay = duration / steps  # Delay between each step

    for angle in angles:
        duty_cycle = angle_to_duty_cycle(angle)
        pwm.ChangeDutyCycle(duty_cycle)
        time.sleep(delay)

def rotate_servo_non_blocking():
    """
    Rotates the servo ±5 degrees and then back to the initial position asynchronously.
    """
    def servo_motion():
        move_servo_smooth("servo1", 75, 120, duration=1.0, steps=50)
        time.sleep(0.3)
        move_servo_smooth("servo1", 120, 75, duration=1.0, steps=50)

    servo_thread = threading.Thread(target=servo_motion)
    servo_thread.start()

# ---------------------- Load TensorFlow Lite Model ----------------------
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input shape
input_shape = input_details[0]['shape']  # e.g., (32, 250, 250, 3)
batch_size = input_shape[0]

# ---------------------- Inference & Control Loop ----------------------
try:
    cap = cv2.VideoCapture(0)
    move("forward", 50)
    print("Rover moving forward...")

    frame_buffer = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame. Exiting...")
            break

        resized_image = cv2.resize(frame, (input_shape[2], input_shape[1]))
        normalized_image = resized_image / 255.0
        frame_buffer.append(normalized_image.astype("float32"))

        if len(frame_buffer) == batch_size:
            input_data = np.array(frame_buffer)
            frame_buffer = []

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            predictions = interpreter.get_tensor(output_details[0]['index'])

            for prediction in predictions:
                class_index = np.argmax(prediction)

                prob_text = f"Weed: {prediction[0]*100:.2f}% | Crop: {prediction[1]*100:.2f}%"
                print(prob_text)

                if class_index == 0:  # Weed detected
                    print("Weed detected! Stopping the rover.")
                    move("stop", 0)
                    time.sleep(0.5)

                    # Trigger servo motion asynchronously
                    print("Activating servo motion.")
                    rotate_servo_non_blocking()

                    # Resume moving forward
                    print("No weed detected. Moving forward...")
                    move("forward", 50)

except KeyboardInterrupt:
    print("Exiting program...")

finally:
    print("Rover stopped, and GPIO cleaned up.")
