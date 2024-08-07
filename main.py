import cv2
import mediapipe as mp
import sounddevice as sd
import numpy as np
import queue
import argparse
import threading
from scipy.signal import sawtooth
from helpers import *

delay_in_seconds = .5
camera = 2 #usually 0 but sometimes needs to be changed. Try values 0-5


def parse_args():
    parser = argparse.ArgumentParser(description="Hand-tracking based audio frequency modulation using Mediapipe and Sounddevice")
    parser.add_argument('-l', '--list-devices', action='store_true', help='show list of audio devices and exit')
    parser.add_argument('frequency', nargs='?', metavar='FREQUENCY', type=float, default=500, help='frequency in Hz (default: %(default)s)')
    parser.add_argument('-d', '--device', type=int_or_str, help='output device (numeric ID or substring)')
    parser.add_argument('-a', '--amplitude', type=float, default=0.2, help='amplitude (default: %(default)s)')
    return parser.parse_args()



def audio_callback(outdata, frames, time, status):
    global start_idx, samplerate, current_frequency, previous_frequency, current_phase, current_amplitude, previous_amplitude, wet_percent, delay_buffer, delay_idx, delay_samples, feedback, lowpass_cutoff, smoothed_lowpass_cutoff, prev_filtered_sample

    try:
        t = (start_idx + np.arange(frames)) / samplerate

        smoothed_frequency = exponential_moving_average(current_frequency, previous_frequency, 0.9)
        smoothed_amplitude = exponential_moving_average(current_amplitude, previous_amplitude, 0.9)

        phase_increment = 2 * np.pi * np.linspace(previous_frequency, smoothed_frequency, frames) / samplerate
        cumulative_phase = current_phase + np.cumsum(phase_increment)
        amplitudes = np.linspace(previous_amplitude, smoothed_amplitude, frames)

        raw_wave = (amplitudes * sawtooth(cumulative_phase)).reshape(-1, 1)

        # Apply delay effect with feedback
        delay_output = np.zeros(frames)
        for i in range(frames):
            current_sample = raw_wave[i]  # Original signal
            delay_sample = delay_buffer[delay_idx]  # Delayed signal

            delay_buffer[delay_idx] = current_sample + feedback * delay_sample  # Update delay buffer with feedback
            delay_output[i] = (1 - wet_percent) * current_sample + wet_percent * delay_sample  # Mix dry and wet signals

            delay_idx = (delay_idx + 1) % delay_samples  # Circular buffer index

        # Smooth the lowpass cutoff frequency
        smoothed_lowpass_cutoff = exponential_moving_average(lowpass_cutoff, smoothed_lowpass_cutoff, 0.9)

        # Calculate the alpha value for the one-pole low-pass filter
        rc = 1.0 / (2 * np.pi * smoothed_lowpass_cutoff)
        dt = 1.0 / samplerate
        alpha = dt / (rc + dt)

        # Apply one-pole low-pass filter
        filtered_output = np.zeros(frames)
        for i in range(frames):
            prev_filtered_sample = one_pole_lowpass(delay_output[i], prev_filtered_sample, alpha)
            filtered_output[i] = prev_filtered_sample

        outdata[:] = filtered_output.reshape(-1, 1)

        current_phase = cumulative_phase[-1] % (2 * np.pi)
        start_idx += frames
        previous_frequency = smoothed_frequency
        previous_amplitude = smoothed_amplitude

    except Exception as e:
        print(f"Error in audio_callback: {e}")

def audio_thread(stop_event):
    global start_idx, samplerate, args, delay_buffer, delay_samples, smoothed_lowpass_cutoff, lowpass_cutoff, prev_filtered_sample
    try:
        samplerate = sd.query_devices(args.device, 'output')['default_samplerate']
        delay_samples = int(samplerate * delay_in_seconds)  # 0.05 seconds delay buffer size
        delay_buffer = np.zeros(delay_samples)
        smoothed_lowpass_cutoff = lowpass_cutoff  # Initialize the smoothed cutoff with the initial cutoff value
        prev_filtered_sample = 0.0  # Initialize the previous filtered sample for the one-pole filter
        with sd.OutputStream(device=args.device, channels=1, callback=audio_callback, samplerate=samplerate, blocksize=1024):
            while not stop_event.is_set():
                stop_event.wait(0.1)
    except Exception as e:
        print(f"{type(e).__name__}: {str(e)}")
    finally:
        print("Audio thread terminating")

def video_thread(frame_queue, stop_event):
    global current_frequency, current_amplitude, wet_percent, feedback, lowpass_cutoff
    # Initialize Mediapipe Hand Detection
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_drawing = mp.solutions.drawing_utils

    # Start webcam capture
    cap = cv2.VideoCapture(camera)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            # Flip the frame horizontally for a mirror effect
            frame = cv2.flip(frame, 1)
            frame_shape = frame.shape

            # Convert the frame color from BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image and detect hands
            results = hands.process(image)

            # Draw hand landmarks if any are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get x and y positions of the index finger (landmark 8)
                    x_index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
                    y_index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

                    x_position = int(x_index_finger * frame_shape[1])
                    y_position = int(y_index_finger * frame_shape[0])

                    # Map the x position to a frequency range (e.g., 200 Hz to 2000 Hz)
                    min_freq = 30.0
                    max_freq = 500.0
                    new_frequency = min_freq + (max_freq - min_freq) * (x_position / frame_shape[1])
                    current_frequency = new_frequency

                    # Map the y position to an amplitude range (e.g., 0.0 to 1.0)
                    min_amp = -0.5
                    max_amp = 1.0
                    new_amplitude = min_amp + (max_amp - min_amp) * (1.0 - (y_position / frame_shape[0]))  # Invert y-axis
                    current_amplitude = new_amplitude

                    # Calculate distance between thumb tip (landmark 4) and pinky joint (landmark 17)
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    pinky_joint = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

                    distance = np.sqrt((thumb_tip.x - pinky_joint.x) ** 2 + (thumb_tip.y - pinky_joint.y) ** 2)  # Euclidean distance

                    # Map this distance to a "wet %" parameter for the delay effect
                    min_distance = 0.0
                    max_distance = 0.3  # Adjust based on testing
                    wet_percent = min(1.0, max(0.0, (distance - min_distance) / (max_distance - min_distance)))

                    # Calculate distance between thumb tip (landmark 4) and index finger base (landmark 5) for feedback control
                    index_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                    feedback_distance = np.sqrt((thumb_tip.x - index_base.x) ** 2 + (thumb_tip.y - index_base.y) ** 2)  # Euclidean distance

                    # Map this distance to a feedback level range (e.g., 0.0 to 0.9)
                    min_feedback_distance = 0.0
                    max_feedback_distance = 0.2  # Adjust based on testing
                    feedback = min(0.9, max(0.0, (feedback_distance - min_feedback_distance) / (max_feedback_distance - min_feedback_distance)))

                    # Calculate distance between fourth finger tip (landmark 16) and its MCP joint (landmark 13)
                    fourth_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                    fourth_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]

                    fourth_distance = np.sqrt((fourth_finger_tip.x - fourth_finger_mcp.x) ** 2 + (fourth_finger_tip.y - fourth_finger_mcp.y) ** 2)  # Euclidean distance

                    # Map this distance to a cutoff frequency range (e.g., 200 Hz to 5000 Hz)
                    min_cutoff_distance = 0.0
                    max_cutoff_distance = 0.2  # Adjust based on testing
                    min_cutoff_freq = 200.0
                    max_cutoff_freq = 5000.0
                    lowpass_cutoff = min_cutoff_freq + (max_cutoff_freq - min_cutoff_freq) * min(1.0, max(0.0, (fourth_distance - min_cutoff_distance) / (max_cutoff_distance - min_cutoff_distance)))

            if not frame_queue.full():
                frame_queue.put(frame)

        print("Video thread terminating")
    finally:
        cap.release()
        hands.close()

if __name__ == '__main__':
    args = parse_args()
    start_idx = 0
    samplerate = None
    current_frequency = args.frequency
    previous_frequency = args.frequency
    current_phase = 0  # Initialize the phase at the beginning
    current_amplitude = args.amplitude  # Initialize amplitude
    previous_amplitude = args.amplitude  # Initialize previous amplitude
    wet_percent = 0.0  # Initialize wet percentage for delay effect
    feedback = 0.0  # Initialize feedback level
    lowpass_cutoff = 500.0  # Initialize lowpass cutoff frequency
    smoothed_lowpass_cutoff = lowpass_cutoff  # Initialize the smoothed cutoff frequency
    prev_filtered_sample = 0.0  # Initialize the previous filtered sample for the one-pole filter
    delay_buffer = None
    delay_samples = 0
    delay_idx = 0

    frame_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    audio_t = threading.Thread(target=audio_thread, args=(stop_event,))
    video_t = threading.Thread(target=video_thread, args=(frame_queue, stop_event))

    audio_t.start()
    video_t.start()

    try:
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                cv2.imshow('Webcam', frame)
            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        stop_event.set()
        audio_t.join()
        video_t.join()
        cv2.destroyAllWindows()