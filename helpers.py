import numpy as np
from scipy.signal import butter, lfilter
import sounddevice as sd

def list_audio_devices():
    print("Available audio devices:")
    print(sd.query_devices())

def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def crossfade_filter_coefficients(prev_b, prev_a, new_b, new_a, alpha):
    return (1 - alpha) * prev_b + alpha * new_b, (1 - alpha) * prev_a + alpha * new_a

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text
    
def exponential_moving_average(current_val, previous_ema, alpha):
    return alpha * current_val + (1 - alpha) * previous_ema

def apply_window(signal, window_type='hamming'):
    window = np.hamming(len(signal))
    return signal * window

def overlap_add(samples, window_len, overlap_len):
    step_len = window_len - overlap_len
    num_steps = (len(samples) - overlap_len) // step_len

    result = np.zeros(num_steps * step_len + window_len)
    window = np.hamming(window_len)
    
    for i in range(num_steps):
        start = i * step_len
        end = start + window_len
        result[start:end] += samples[start:end] * window

    return result[:len(samples)]

def one_pole_lowpass(x, prev_y, alpha):
    return alpha * x + (1 - alpha) * prev_y