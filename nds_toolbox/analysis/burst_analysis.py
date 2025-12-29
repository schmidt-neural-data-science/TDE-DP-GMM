import numpy as np
from nds_toolbox.preprocessing.filters import gaussian_bandpass_filter
from nds_toolbox.preprocessing.features import amplitude_envelope
from nds_toolbox.analysis.utils import safe_corrcoef
from neurodsp.spectral import compute_spectrum



def thresholding_bursts(signal, detect_k=1.5, width_k=1.0):
    """
    Detect bursts using a dual threshold:
      - Detection threshold: median + detect_k * std.
      - Burst width threshold: median + width_k * std.

    Parameters
    ----------
    signal : array_like
        The amplitude envelope.
    detect_k : float, optional
        Multiplier for the detection threshold (default 1.5).
    width_k : float, optional
        Multiplier for the burst width threshold (default 1.0).

    Returns
    -------
    final_burst_index : ndarray
        A binary array (1: burst, 0: no burst).

    Notes:
        The algorithm is inspired by NeuroDSP's dual-threshold burst detection [1].
        For the original method, see [2].

    References:
    [1] Cole, S., Donoghue, T., Gao, R., & Voytek, B. (2019). NeuroDSP: A package for
neural digital signal processing. Journal of Open Source Software, 4(36), 1272.
DOI: 10.21105/joss.01272
    [2] Feingold, J., Gibson, D. J., DePasquale, B., & Graybiel, A. M. (2015).
           Bursts of beta oscillation differentiate postperformance activity in
           the striatum and motor cortex of monkeys performing movement tasks.
           Proceedings of the National Academy of Sciences, 112(44), 13687–13692.
           DOI: https://doi.org/10.1073/pnas.1517629112
    """
    med, std = np.median(signal), np.std(signal)
    detect_thresh = med + detect_k * std
    width_thresh = med + width_k * std

    # Find initial burst onsets/offsets using the detection threshold
    above_detect = signal > detect_thresh
    burst_starts = np.where(np.diff(above_detect.astype(int)) == 1)[0]
    burst_ends = np.where(np.diff(above_detect.astype(int)) == -1)[0]

    # Include edges if needed
    if above_detect[0]:
        burst_starts = np.insert(burst_starts, 0, 0)
    if above_detect[-1]:
        burst_ends = np.append(burst_ends, len(signal) - 1)

    final_burst_index = np.zeros_like(signal, dtype=int)

    # For each burst, extend boundaries using the width threshold
    for start, end in zip(burst_starts, burst_ends):
        left = start
        while left > 0 and signal[left - 1] >= width_thresh:
            left -= 1
        right = end
        while right < len(signal) - 1 and signal[right + 1] >= width_thresh:
            right += 1
        final_burst_index[left:right + 1] = 1

    return final_burst_index


def optimize_threshold_params(signal, filter_freq, fs, fwhm=5.0,
                              detect_k_range=np.arange(0., 3.1, 0.1),
                              width_k_range=np.arange(0., 3.1, 0.1)):
    """
    Optimize threshold parameters for burst detection by maximizing the Pearson correlation
    between the detected bursts and the amplitude envelope of the signal.
    """
    # Precompute the filtered signal and its envelope.



    filtered_signal = gaussian_bandpass_filter(signal, filter_freq, fs, fwhm=fwhm)
    enveloped_signal = amplitude_envelope(filtered_signal)

    # Define a helper function that computes the correlation for given parameters.
    def compute_corr(detect_k, width_k):
        detected_bursts = thresholding_bursts(enveloped_signal, detect_k=detect_k, width_k=width_k)

        return safe_corrcoef(detected_bursts, enveloped_signal)

    # Create a grid of parameter values.
    detect_grid, width_grid = np.meshgrid(detect_k_range, width_k_range, indexing='ij')

    # Vectorize the helper function to apply it over the grid.
    vectorized_compute_corr = np.vectorize(compute_corr)
    corr_mat = vectorized_compute_corr(detect_grid, width_grid)

    # Find the indices of the best correlation value.
    best_corr_idx = np.unravel_index(np.argmax(corr_mat), corr_mat.shape)
    best_detect_k_corr = detect_k_range[best_corr_idx[0]]
    best_width_k_corr = width_k_range[best_corr_idx[1]]
    best_corr = corr_mat[best_corr_idx]

    return enveloped_signal,  (best_detect_k_corr, best_width_k_corr), best_corr,


def state_lifetimes(states, num_states = None, fs=None):

    if num_states is None:
        num_states = states.max()+1

    if fs is None:
        fs = 1

    states = np.array(states)

    change_pos = np.flatnonzero(np.diff(states)) + 1  # indices after each a state change
    starts = np.r_[0, change_pos]
    ends = np.r_[change_pos, len(states)]

    lifetime_all = (ends - starts) / fs
    lifetime_states = [lifetime_all[states[starts] == i] for i in range(num_states)]
    return lifetime_states

def mean_state_lifetimes(states, num_states = None, fs=None):
    if num_states is None:
        num_states = states.max()+1

    if fs is None:
        fs = 1

    lt_states = state_lifetimes(states, num_states, fs)
    mean_lt_states = np.array([np.nanmean(lt) if lt.size else np.nan for lt in lt_states])
    return mean_lt_states


def state_intervals(states, num_states = None, fs = None):

    if num_states is None:
        num_states = states.max()+1

    if fs is None:
        fs = 1

    states = np.array(states)
    change_pos = np.flatnonzero(np.diff(states))+1
    starts = np.r_[0, change_pos]
    ends = np.r_[change_pos, len(states)]

    state_intervals = []
    for s in np.arange(num_states):
        id = np.where(states[starts] == s)[0] # get start positions for a state and unpack tuple

        #intervals: gap between next start and the end
        intervals = starts[id[1:]] - ends[id[:-1]]

        intervals = np.nan_to_num(intervals, nan=0.0)
        state_intervals.append(np.array(intervals) / fs)

    return state_intervals

def mean_state_intervals(states, num_states = None, fs = None):
    if num_states is None:
        num_states = states.max()+1

    if fs is None:
        fs = 1

    intervals_states = state_intervals(states, num_states= num_states, fs = fs)
    return np.array([np.nanmean(intervals) for intervals in intervals_states])

def CV_state_intervals(states, num_states = None, fs = None):
    """
    compute the coefficient of variation (CV = std/mean) of state intervals

    notes:
    - CV = 0 -> Perfectly periodic — every interval is identical.
    - CV ≈ 1 -> Matches an exponential (Poisson) distribution; intervals are as variable as their mean.
    - CV > 1 -> Intervals are highly uneven — long gaps alternate with clusters of short intervals (bursty).

    references:

    :param states:
    :param num_states:
    :param fs:
    :return:
    """

    if num_states is None:
        num_states = states.max()+1

    if fs is None:
        fs = 1

    intervals_states = state_intervals(states, num_states= num_states, fs = fs)
    return np.array([np.nanstd(intervals) / np.nanmean(intervals) for intervals in intervals_states])








def state_burst_rates(states, num_states = None, fs = None):

    if num_states is None:
        num_states = states.max()+1

    if fs is None:
        fs = 1


    # burst starts = indices where state changes or at t=0
    change = np.r_[True, np.diff(states) != 0]
    starts = np.where(change)[0]
    start_states = states[starts]


    recording_time = len(states) / fs
    counts = np.array([np.nansum(start_states == i) for i in range(num_states)])
    return counts / recording_time


def state_fractional_occupancies(states, num_states = None):
    if num_states is None:
        num_states = states.max()+1
    states_onehot = np.eye(num_states)[states]
    return np.mean(states_onehot, axis = 0)



from scipy.signal import welch
def compute_state_psd(signal, states, fs, num_states = None):
    if num_states is None:
        num_states = states.max()+1

    psds = []
    f_ref = []
    for i in range(num_states):

        if i not in np.unique(states):
            psds.append(np.array([]))
            f_ref.append(np.array([]))
            continue

        signal_states = signal[states == i]
        f, psd = welch(signal_states, fs,scaling = "density", average = "mean", nperseg=fs*2) # unit = V**2/Hz

        f_ref.append(f)
        psds.append(psd)
    return f_ref, psds


def state_powers(signal, states, fs, num_states = None):
    """
    compute area under psd for each state
    """
    if num_states is None:
        num_states = states.max()+1

    powers = np.zeros(num_states)
    for i in range(num_states):

        if i not in np.unique(states):
            continue

        x = signal[states == i]
        x0 = x - np.mean(x) #remove DC offset
        powers[i] = np.var(x0)

    return powers


# this has to be modified later
from scipy.signal import find_peaks
def peak_frequency(signal, fs, prom_frac = 0.3, min_freq_dist = 4, powerlaw_correction=True):
    f, psd = compute_spectrum(signal, fs)
    if powerlaw_correction:
        psd = f * psd #supress the power of low frequency

    df = f[1] - f[0]
    min_prom = prom_frac * psd.ptp()
    distance_bins = max(1, int(min_freq_dist / df))

    peaks, _ = find_peaks(psd,  prominence = min_prom, distance = distance_bins)
    if peaks.size == 0:
        return np.nan
    else:
        return f[peaks]

#this also
def state_peak_frequencies(signal, states, fs, num_states = None, method="argmax", powerlaw_correction=True):
    if num_states is None:
        num_states = states.max()+1

    peak_freqs = []
    for i in range(num_states):
        signal_states = signal[states == i]
        peak_freqs.append(peak_frequency(signal_states, fs, method = method, powerlaw_correction = powerlaw_correction))
    return np.array(peak_freqs)





def summarize_states(signal, states, fs, num_states = None):
    if num_states is None:
        num_states = states.max()+1

    #state occurrence rate
    br_states = state_burst_rates(states, fs = fs, num_states = num_states)

    lt_states = state_lifetimes(states, fs = fs, num_states = num_states)
    #state mean lifetime
    median_lt_states = mean_state_lifetimes(states, fs = fs, num_states = num_states)

    #state fractional occupancy
    fo_states = state_fractional_occupancies(states, num_states = num_states)

    #state power spectra
    f, psds = compute_state_psd(signal, states, fs, num_states = num_states)

    #state power
    power_states = state_powers(signal, states, fs, num_states = num_states)

    return {
        "fractional_occupancies": fo_states,
        "lifetimes": lt_states,
        "median_lt_states": median_lt_states,
        "intervals": state_intervals(states, fs = fs, num_states = num_states),
        "mean_intervals": mean_state_intervals(states, fs = fs, num_states = num_states),
        "CV_intervals": CV_state_intervals(states, fs = fs, num_states = num_states),
        "burst_rates": br_states,
        "powers": power_states,
        "spectra": (f, psds),
    }
