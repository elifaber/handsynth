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

The following hand gestures control different audio parameters:

1. Frequency Control: Move your index finger horizontally to change the frequency (30 Hz to 500 Hz).
2. Amplitude Control: Move your index finger vertically to change the amplitude (note that 0 lies somewhere in the middle of the screen due to needing your whole hand on the screen.)
3. Delay Wet Percentage: The distance between your thumb tip and pinky base controls the wet percentage of the delay effect.
4. Feedback Level: The distance between your thumb tip and index finger base controls the feedback level of the delay effect.
5. Low-Pass Cutoff Frequency: The distance between your fourth finger tip and its base controls the cutoff frequency of the low-pass filter.


### Troubleshooting

- Webcam Not Detected: Ensure your webcam is properly connected and recognized by your system. If necessary, change the camera variable in the script to a different value (e.g., 0-5) to select the correct camera.
- Audio Issues: Ensure your audio output device is correctly specified. Use the --list-devices option to identify the correct device ID.

### Dependencies
opencv-python
mediapipe
sounddevice
numpy
scipy

Install these with:

```bash
pip install opencv-python mediapipe sounddevice numpy scipy
