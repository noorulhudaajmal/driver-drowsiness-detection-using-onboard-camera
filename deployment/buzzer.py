import RPi.GPIO as GPIO
import time

# GPIO pins
BUZZER_PIN = 19
SENSOR_PIN = 17

def setup():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUZZER_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(SENSOR_PIN, GPIO.IN)


def detect_accident():
    """
    Detects if an accident has occurred based on the sensor input.
    """
    sensor_value = GPIO.input(SENSOR_PIN)
    if sensor_value == GPIO.HIGH:
        return True
    return False


def trigger_buzzer(accident_detected):
    """
    Controls the buzzer based on accident detection.
    """
    if accident_detected:
        GPIO.output(BUZZER_PIN, GPIO.HIGH)
    else:
        GPIO.output(BUZZER_PIN, GPIO.LOW)


def main():
    setup()

    while True:
        accident_detected = detect_accident()
        trigger_buzzer(accident_detected)
        time.sleep(0.1)  # delay


# if __name__ == '__main__':
#     try:
#         main()
#     except KeyboardInterrupt:
#         GPIO.cleanup()
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         GPIO.cleanup()
