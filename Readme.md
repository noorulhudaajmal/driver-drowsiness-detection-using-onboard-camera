# Driver Drowsiness Detection using onboard Camera

## Overview

This project implements a real-time Driver Drowsiness Detection System using machine learning models optimized for deployment on edge devices like the Raspberry Pi 4. The system captures live video input from an infrared camera to detect signs of driver drowsiness and alerts the driver through a buzzer if drowsiness is detected. Additionally, the system detects accidents using an accelerometer and captures footage leading up to the event for analysis.

## Core Features

### 1. Drowsiness Detection
- **Real-time Monitoring:** The system continuously monitors the driver's face using an infrared camera mounted on dashboard.
- **Facial Landmark Detection:** Key facial features, such as the eyes and mouth, are detected using the `Dlib` library.
- **Drowsiness Detection:** The system uses CNNs to analyze the extracted features for signs of drowsiness, such as blinking and yawning.
- **Alert Mechanism:** If drowsiness is detected (based on the `PERCLOS` algorithm), the system triggers a buzzer to alert the driver. The buzzer remains active until the driver regains alertness (or turns off manually).

### 2. Accident Detection
- **Accelerometer Monitoring:** An `MPU6050` accelerometer monitors the vehicle for sudden deceleration, indicating a potential accident.
- **Footage Capture:** The system automatically saves the last five seconds of video footage before an accident is detected, which can be uploaded to a web portal for further analysis (analyzing and improving model if accident is caused by non-detected drowsy behaviour of the driver).

### 3. Web Portal
- **Model Updates:** Users can download the latest versions of the drowsiness detection models from the web portal.
- **Accident Footage Upload:** Users can upload accident footage, along with comments, for developers to analyze and improve the detection models.

## System Components

### Hardware
- **Raspberry Pi 4:** The main processing unit for the system.
- **Infrared Camera:** Captures the driver's face in various lighting conditions, including complete darkness.
- **MPU6050 Accelerometer:** Detects sudden deceleration to identify potential accidents.
- **Buzzer:** Alerts the driver when drowsiness is detected.

### Software
- **TensorFlow Lite:** Used to optimize and run machine learning models on the Raspberry Pi.
- **OpenCV:** Handles video frame processing, including resizing and converting images to grayscale.
- **Dlib:** Provides facial landmark detection to extract features like the eyes and mouth.
- **Picamera:** Captures video frames from the Raspberry Pi camera.
- **Django:** Web framework for the web portal.

## Directory Structure

- **training/**
   - Jupyter notebooks for model training.
   - Logic functions for facial detection and feature extraction.
   - Python scripts for converting TensorFlow models to TensorFlow Lite.

- **deployment/**
   - Contains the main application code, including drowsiness detection, accident detection, and alert logic.

- **webapp/**
   - Django application for the web portal.

## Installation and Setup

### Hardware Setup
1. Assemble the Raspberry Pi, Infrared Camera, and MPU6050 Accelerometer.
2. Connect the Buzzer to the appropriate GPIO pins on the Raspberry Pi.

### Software Setup
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Clone the repository:
   ```bash
   git clone https://github.com/noorulhudaajmal/driver-drowsiness-detection-using-onboard-camera.git
   ```
3. For webapp, navigate to the project directory:
   ```bash
   cd webapp
   django-admin startproject driveGuard_Solutions
   cd driveGuard_Solutions
   python manage.py runserver
   ```

### Running the System
1. Start the Drowsiness Detection System:
   ```bash
   python deployment/driver_monitor.py
   ```

## Contributions

Feel free to contribute to the project by submitting issues or pull requests.

---
