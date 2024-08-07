# handsynth

This project uses hand-tracking with Mediapipe to modulate audio frequency, create delay and feedback effects, and apply a low-pass filter in real-time. The parameters of these effects are controlled by different aspects of hand gestures detected through the webcam.

## Table of Contents

- [Running the Code](#running-the-code)
- [Usage](#usage)
- [Controls](#controls)
- [Troubleshooting](#troubleshooting)
- [Dependencies](#dependencies)


### Prerequisites

1. Python 3.6 or higher
2. Webcam
3. Compatible audio output device

### Installing Dependencies

First, make sure you have `pip` installed. Then, install the required packages by running the following command:

```bash
pip install opencv-python mediapipe sounddevice numpy scip
