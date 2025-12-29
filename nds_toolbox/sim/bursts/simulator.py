"""
The following burst simulator is inspired from:
- "Quinn2019_BurstHMM/hmm_util_get_simulation.m"
- "Cho2022_BurstDetection/utils/util_data/generates_simulaiton.m"
- Cole, S., Donoghue, T., Gao, R., & Voytek, B. (2019). NeuroDSP: A package for
neural digital signal processing. Journal of Open Source Software, 4(36), 1272.
DOI: 10.21105/joss.01272
"""


import numpy as np
from scipy.signal import sawtooth
from scipy.signal.windows import tukey
from neurodsp.filt import filter_signal





def _get_duration(duration_param, rng):
    """
    Determines a duration value.

    If duration_param is a scalar (int or float), returns it.
    If it is a two-element list/tuple/array, returns a random value drawn uniformly
    between the two values.
    """
    if isinstance(duration_param, (list, tuple, np.ndarray)) and len(duration_param) == 2:
        return rng.uniform(duration_param[0], duration_param[1])
    elif np.isscalar(duration_param):
        return duration_param
    else:
        raise ValueError("Duration parameter must be a scalar or a two-element list/tuple/array.")




def _simulate_bursts(time_vec, fs, f, burst_cycles, noise_duration,
                           burst_type, burst_amp_sigma=0.1, chi=0.15,
                           use_tukey=True, tukey_alpha=0.25, power_law_scale = True, rng = None):
    """
    Simulates a signal over a given time vector where each segment is chosen to be either noise or a burst.
    Burst segments are modulated such that each cycle of the oscillation is scaled by its own random amplitude.

    State 0 is reserved for noise (its segments are left as zeros so that noise can be added later),
    while dstates > 0 produce burst segments.

    In this version the burst duration is specified as the number of cycles.
    For each burst, the actual duration in seconds is computed as:
         duration = (number of cycles) / (burst frequency)

    This implementation forces an integer number of cycles by computing the number of samples per cycle.

    Burst and noise durations can be provided as fixed values (scalar) or as a two-element
    sequence [min, max] to sample a random value uniformly (for noise).

    Parameters:
        time_vec (array-like): The time points for the simulation.
        fs (float): The sampling frequency.
        f (scalar or array-like): Frequency (in Hz) for burst segments.
            - If a scalar, all burst states use the same frequency.
            - If array-like, its length must equal the number of burst states.
        burst_cycles (scalar or two-element sequence): Number of cycles per burst.
            If a two-element sequence, a random number of cycles is drawn uniformly from that range.
        noise_duration (scalar or two-element sequence): Duration (in seconds) for each noise segment.
            If a two-element sequence, a random duration is drawn uniformly from that interval.
        burst_type (str): "sine" or "sawtooth" determines the waveform for burst segments.
        chi (float): Exponent controlling the degree of power-law scaling. The default value was obtained from "Quinn2019_BurstHMM/hmm_util_get_simulation.m"
        burst_amp_sigma (float): Standard deviation for the normal distribution used to generate
                           the random amplitude per cycle.
        use_tukey (bool): Whether to apply a Tukey window to each burst.
        tukey_alpha (float): Alpha parameter for the Tukey window (0 < alpha < 1). Controls
                             how aggressively the window tapers.

    Returns:
        signal (ndarray): The simulated signal (same length as time_vec). Burst segments contain oscillations;
                          noise segments are left as zeros.
        state_ts (ndarray): A one-hot encoded state vector of shape (len(time_vec),).
                            Values 0 = noise, 1..num_data = burst states.
    """


    if rng is None:
        rng = np.random.default_rng()



    # Convert f to a numpy array for convenience.
    freqs_array = np.array(f)
    freqs_size = freqs_array.size

    if freqs_size == 1:
        use_scalar_freq = True
    else:
        use_scalar_freq = False

    # Define the number of states.
    # State 0 is noise, and states 1, 2, ... are burst segments.
    num_states = freqs_size + 1
    init_p = np.ones(num_states) / num_states  # Uniform probabilities.

    signal = np.zeros(len(time_vec))
    states = np.zeros(len(time_vec))

    t = 0.0  # Global time pointer (in seconds)
    last_state = None

    while t < time_vec[-1]:
        # Choose a state that is not the same as the last state.
        if last_state is None:
            current_state = rng.choice(np.arange(num_states), p=init_p)
        else:
            valid_states = np.delete(np.arange(num_states), last_state)
            valid_probs = np.delete(init_p, last_state)
            valid_probs = valid_probs / valid_probs.sum()
            current_state = rng.choice(valid_states, p=valid_probs)

        last_state = current_state
        start_idx = int(t * fs)

        if current_state == 0:
            # --- Noise Segment ---
            curr_noise_duration = _get_duration(noise_duration, rng)
            end_idx = start_idx + int(np.round(curr_noise_duration * fs))
            if end_idx > len(time_vec):
                end_idx = len(time_vec)
            states[start_idx:end_idx] = 0  # Noise state.
            t += curr_noise_duration
        else:
            # --- Burst Segment ---
            if use_scalar_freq:
                freq_burst = freqs_array.item()
            else:
                # For burst states, use current_state - 1 since state 0 is noise.
                freq_burst = freqs_array[current_state - 1]

            # Determine the number of cycles
            if isinstance(burst_cycles, (list, tuple, np.ndarray)) and len(burst_cycles) == 2:
                num_cycles = rng.integers(burst_cycles[0], burst_cycles[1]+1)

            elif np.isscalar(burst_cycles):
                num_cycles = int(burst_cycles)
            else:
                raise ValueError("burst_cycles must be a scalar or a two-element sequence.")

            # Compute the number of samples per cycle and then the total samples.
            samples_per_cycle = int(round(fs / freq_burst))
            total_samples = samples_per_cycle * num_cycles
            aligned_time = np.arange(total_samples) / fs


            burst_amplitudes = np.abs(rng.normal(1, burst_amp_sigma))
            if not use_scalar_freq:
                if power_law_scale:
                # 1/f scaling based on burst frequency (when there are multiple frequency components, this scaling matters)
                    power_law = 1/ (freq_burst ** chi)
                    burst_amplitudes *= power_law



            # Generate the burst waveform.
            if burst_type == "sine":
                burst_signal = burst_amplitudes * np.sin(2 * np.pi * freq_burst * aligned_time)
            elif burst_type == "sawtooth":
                burst_signal = burst_amplitudes * sawtooth(2 * np.pi * freq_burst * aligned_time, width=1)
            else:
                raise ValueError("Unsupported burst_type. Choose 'sine' or 'sawtooth'.")

            # Apply a Tukey window if requested
            if use_tukey:
                w = tukey(total_samples, alpha=tukey_alpha)
                burst_signal *= w

            end_idx = start_idx + total_samples
            if end_idx > len(time_vec):
                # Truncate burst if it exceeds the signal length.
                burst_signal = burst_signal[:len(time_vec) - start_idx]
                end_idx = len(time_vec)

            signal[start_idx:end_idx] = burst_signal
            states[start_idx:end_idx] = current_state

            # Advance time exactly by the burst duration.
            curr_burst_duration = total_samples / fs
            t += curr_burst_duration


    return np.array(signal), np.array(states).astype(int)

def _generate_colored_noise(num_data, fs, beta=1, rng=None):
    """
    Generate colored noise with a 1/f^beta power spectrum.

    Parameters:
        beta (float): Exponent for the 1/f^beta distribution.
                      Use beta=0 for white noise, beta=1 for pink noise, beta=2 for brown noise.
        num_data (int): Number of samples in the noise signal.

    Returns:
        np.ndarray: The generated noise signal.

    Notes:
        - This function is based on neurodsp.sim.aperiodic.sim_powerlaw.[1]
        - The original reference is [2]

    References:
        [1] Cole, S., Donoghue, T., Gao, R., & Voytek, B. (2019). NeuroDSP: A package for
neural digital signal processing. Journal of Open Source Software, 4(36), 1272.
DOI: 10.21105/joss.01272. https://neurodsp-tools.github.io/neurodsp/_modules/neurodsp/sim/aperiodic.html#sim_powerlaw
        [2] Timmer, J., & Konig, M. (1995). On Generating Power Law Noise.
           Astronomy and Astrophysics, 300, 707–710.
    """

    if rng is None:
        rng = np.random.default_rng()

    # Generate white noise in the time domain.
    white_noise = rng.standard_normal(num_data)

    # Transform to frequency domain using the real FFT.
    spectrum = np.fft.rfft(white_noise)

    # Create frequency array; np.fft.rfftfreq returns frequencies for the rFFT.
    f = np.fft.rfftfreq(num_data, 1 / fs)

    spectrum[1:] /= f[1:] ** (beta / 2)  # |spectrum| ∝ 1/f^(β/2) (hence, power ∝ 1/f^β)
    spectrum[0] = 0

    # Transform back to the time domain.
    colored_noise = np.fft.irfft(spectrum, n=num_data).real
    #colored_noise = (colored_noise - colored_noise.mean()) / colored_noise.std()

    return np.array(colored_noise)


def _add_noise(bursts, states, noise, snr_db, use_filter = True, fs = None, highpass_f = 0.5):
    """
    Mix a clean burst signal with noise to achieve a target SNR (dB).

    Parameters
    ----------
    bursts : 1-D array
        Array that already contains bursts *and* zeros where there is noise.
    states : 1-D int array
        Parallel state vector (0 = noise gaps, >0 = burst indices).
    noise : 1-D array
        Noise signal (same length as bursts).
    snr_db : float
        Desired SNR, in decibels, defined as
            10·log10( signal_power / noise_power ).

    Returns
    -------
    noisy : 1-D array
        bursts + scaled_noise
    """

    if bursts.shape != noise.shape or bursts.shape != states.shape:
        raise ValueError("bursts, noise, and states must have identical shapes")
    if use_filter and (fs is None or highpass_f is None):
        raise ValueError("`fs` and `filter_frequency` must be provided when use_filter=True")


    bursts_copy = bursts.copy()
    noise_copy = noise.copy()  # this will be scaled based on SNR below
    all_states = np.unique(states)

    if use_filter:
        filtered_noise = filter_signal(noise_copy, fs, 'highpass', f_range= (highpass_f, None), remove_edges = False)

    else:
        filtered_noise = noise_copy

    noise_power = np.var(filtered_noise)
    for s in all_states:

        mask = states == s

        if s == 0:
            burst_power = np.var(bursts_copy[states != 0])

        else:
            burst_power = np.var(bursts_copy[mask])


        # desired noise power for that SNR
        snr_linear = 10 ** (snr_db / 10)
        desired_burst_power = snr_linear * noise_power

        # Scale bursts
        scaling_factor = np.sqrt(desired_burst_power / burst_power)
        bursts_copy[mask] *= scaling_factor

    return bursts_copy + noise_copy, bursts_copy


def old_add_noise(bursts, states, noise, snr_db):
    """
    Mix a clean burst signal with noise to achieve a target SNR (dB).

    Parameters
    ----------
    bursts : 1-D array
        Array that already contains bursts *and* zeros where there is noise.
    states : 1-D int array
        Parallel state vector (0 = noise gaps, >0 = burst indices).
    noise : 1-D array
        Noise signal (same length as bursts).
    snr_db : float
        Desired SNR, in decibels, defined as
            10·log10( signal_power / noise_power ).

    Returns
    -------
    noisy : 1-D array
        bursts + scaled_noise
    """
    # power of the *non-zero* (burst) part of the signal
    signal_power = np.mean(bursts[states != 0] ** 2)

    # desired noise power for that SNR
    snr_lin            = 10 ** (snr_db / 10)
    desired_noise_power = signal_power / snr_lin

    scaled_noise        = noise * np.sqrt(desired_noise_power)

    return bursts + scaled_noise


def simulate_bursty_signal(
        time_vec,
        fs,
        freq,
        burst_cycles_param,
        noise_duration_param,
        burst_type="sine",
        use_filter = True,
        highpass_f = 0.5,
        snr_db=0,
        burst_amp_sigma=0.1,
        beta=1,
        chi=0.15,
        use_tukey=True,
        tukey_alpha=0.25,
        rng=None):
    """
    Simulate a bursty signal with added noise.

    """

    if rng is None:
        rng = np.random.default_rng()

    # Generate bursts using the specified burst type and amplitude scale.
    bursts, states = _simulate_bursts(
        time_vec, fs, freq,
        burst_cycles_param, noise_duration_param,
        burst_type=burst_type, burst_amp_sigma=burst_amp_sigma, chi=chi,
        use_tukey=use_tukey, tukey_alpha=tukey_alpha, rng=rng
    )

    # Generate colored noise
    noise = _generate_colored_noise(len(time_vec), fs, beta, rng=rng)

    # scale noise based on the desired SNR
    signal, scaled_bursts = _add_noise(bursts, states, noise, snr_db, use_filter = use_filter, fs = fs, highpass_f = highpass_f)


    return {"signal": signal,
            "states": states,
            "bursts": scaled_bursts,
            "noise": noise,}
