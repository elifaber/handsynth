# handsynth

This project uses hand-tracking with Mediapipe to modulate audio frequency, create delay and feedback effects, and apply a low-pass filter in real-time. The parameters of these effects are controlled by different aspects of hand gestures detected through the webcam.

## Table of Contents

- [Controls](#controls)
- [Troubleshooting](#troubleshooting)
- [Dependencies](#dependencies)


### Prerequisites

1. Python 3.6 or higher
2. Webcam
3. Compatible audio output device

### Installing Dependencies

First, make sure you have `pip` installed. Then, install the required packages by running the following command:

pip install opencv-python mediapipe sounddevice numpy scipy

### Controls

Controls
The following hand gestures control different audio parameters:

1. Frequency Control: Move your index finger (landmark 8) horizontally to change the frequency (30 Hz to 500 Hz).
2. Amplitude Control: Move your index finger vertically to change the amplitude (-0.5 to 1.0).
3. Delay Wet Percentage: The distance between your thumb tip (landmark 4) and pinky joint (landmark 17) controls the wet percentage of the delay effect.
4. Feedback Level: The distance between your thumb tip (landmark 4) and index finger base (landmark 5) controls the feedback level of the delay effect.
5. Low-Pass Cutoff Frequency: The distance between your fourth finger tip (landmark 16) and its MCP joint (landmark 13) controls the cutoff frequency of the low-pass filter.