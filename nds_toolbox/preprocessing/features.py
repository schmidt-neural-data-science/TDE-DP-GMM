import numpy as np
import pywt

def compute_wavelet(data, fs, f_min = 1, f_max = 50, df = 0.5, wavelet = "cmor1.5-1.0" ):
    dt = 1/fs
    t = np.arange(len(data))*dt
    f = np.arange(f_min, f_max+df, df)

    fc = pywt.central_frequency(wavelet)
    scales = fc / (f * dt)

    # Compute CWT
    coef, _ = pywt.cwt(data, scales, wavelet, sampling_period=dt)
    amp = np.abs(coef)
    return t, f, amp



# Time-delay embeddings
def compute_tde(data, num_embeddings = None, delay=1, verbose=True):
    if num_embeddings is None:
        raise ValueError("num_embeddings must be provided.")
    if num_embeddings % 2 == 0:
        raise ValueError("num_embeddings should be odd for symmetric embedding.")
    if delay < 1:
        raise ValueError("delay must be >= 1.")

    data = np.asarray(data)
    half_window = num_embeddings // 2
    span = half_window * delay

    if len(data) < 2 * span + 1:
        raise ValueError("Not enough data for the requested num_embeddings and delay.")

    segments = []

    # embeddings
    for i in range(span, len(data) - span):
        segment = data[i - span: i + span + 1: delay]
        segments.append(segment)

    segments = np.asarray(segments)

    if verbose:
        print(f"{span} data points were lost from both ends.")
        print("data shape:", segments.shape)

    return np.array(segments)


def choose_embedding_dim(freq, fs, min_cycles = 2.5, ensure_odd = True, verbose = True):
    """Choose an embedding dimension to cover at least {min_cycles} cycles"""

    if freq <= 0 or fs <= 0:
        raise ValueError("freq and fs must be positive")

    min_window  = min_cycles/freq
    num_emb = int(np.ceil(min_window * fs))

    if ensure_odd and num_emb % 2 == 0:
        num_emb += 1

    if verbose:
        print(f"Chosen window size = {np.round(min_window, 3)} (sec): minimum cycles of {min_cycles} at {freq} Hz")
        print(f"Chosen embedding dimension = {num_emb}: sampling rate = {fs} Hz")

    return num_emb


# trimming data to get the same number of data points after TDE
def trim_data(data, num_embeddings, delay=1, verbose=True):
    half_data_window = num_embeddings // 2
    half_data_lost = delay * half_data_window
    trimmed_data = data[half_data_lost:-half_data_lost]

    if verbose:
        print(f"Lost {half_data_lost} data points were lost from both ends.")
        print("data shape:", np.array(trimmed_data).shape)

    return trimmed_data




from scipy.signal import hilbert

def amplitude_envelope(signal):
    return np.abs(hilbert(signal))