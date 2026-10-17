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
from scipy.ndimage import gaussian_filter1d
from neurodsp.sim import sim_oscillation





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


def make_transition_matrix(num_states, state_transition, trans_mat=None):
    if num_states < 2:
        raise ValueError("num_states must be at least 2.")

    if state_transition == "custom":
        if trans_mat is None:
            raise ValueError("trans_mat must be provided when state_transition='custom'.")
        trans_mat = np.asarray(trans_mat, dtype=float)

        if trans_mat.shape != (num_states, num_states):
            raise ValueError(
                f"trans_mat must have shape ({num_states}, {num_states}), got {trans_mat.shape}."
            )

        if not np.allclose(trans_mat.sum(axis=1), 1.0):
            raise ValueError("Each row of trans_mat must sum to 1.")

    elif state_transition == "uniform":
        trans_mat = np.ones((num_states, num_states), dtype=float) / num_states

    elif state_transition == "uniform_except_self":
        trans_mat = np.ones((num_states, num_states), dtype=float)
        np.fill_diagonal(trans_mat, 0.0)
        trans_mat /= (num_states - 1)

    elif state_transition == "return_to_baseline":
        trans_mat = np.zeros((num_states, num_states), dtype=float)
        trans_mat[0, 1:] = 1.0 / (num_states - 1)  # baseline -> others
        trans_mat[1:, 0] = 1.0                     # others -> baseline

    else:
        raise ValueError(
            "state_transition must be one of: "
            "'custom', 'uniform', 'uniform_except_self', 'return_to_baseline'."
        )

    return trans_mat

import numpy as np

from scipy.signal import sawtooth
from scipy.signal.windows import tukey
from scipy.ndimage import gaussian_filter1d

from neurodsp.sim import sim_oscillation


def _simulate_bursts(
    time_vec,
    fs,
    f,
    burst_cycles,
    noise_duration,
    burst_type="sine",
    state_transition="return_to_baseline",
    transition_matrix=None,
    burst_amp_sampling="absolute_normal",
    burst_amp_factors=None,
    burst_amp_sigma=0.1,
    chi=0.15,
    use_tukey=True,
    tukey_alpha=0.25,
    rise_ratio=0.3,
    rise_ratio_sampling="uniform",
    sawtooth_width=0.7,
    sawtooth_smooth_fraction=0.06,
    sawtooth_width_sampling="uniform",
    power_law_scale=False,
    rng=None,
):
    """
    Generate a signal containing transient oscillatory bursts separated by
    baseline segments.

    State 0 represents the baseline (noise) state, while states 1 through K
    represent oscillatory burst states. The baseline segments are initialized
    to zero; background noise must be added separately.

    Parameters
    ----------
    time_vec : array-like
        Time vector defining the number of samples in the simulation.
        Sampling is assumed to be uniform at the specified sampling frequency.

    fs : float
        Sampling frequency in Hz.

    f : float or array-like
        Oscillation frequencies in Hz.

        - If scalar, generate a single burst state at the specified frequency.
        - If array-like, each frequency defines a distinct burst state.

    burst_cycles : int or tuple of int
        Number of oscillation cycles per burst.

        - If scalar, use a fixed number of cycles.
        - If a length-2 sequence, sample an integer uniformly between
          the specified minimum and maximum (inclusive).

    noise_duration : float or tuple of float
        Duration of baseline segments in seconds.

        - If scalar, use a fixed duration.
        - If a length-2 sequence, sample uniformly between the
          specified minimum and maximum.

    burst_type : {"sine", "asine", "sawtooth"}, default="sine"
        Waveform used to generate oscillatory bursts.

        - "sine":
          Standard sinusoidal oscillation.

        - "asine":
          Asymmetric sinusoidal oscillation with controllable
          rise-decay symmetry. Generated using NeuroDSP's
          `sim_oscillation` function with `cycle="asine"`.

        - "sawtooth":
          Sawtooth oscillation with adjustable asymmetry and
          optional Gaussian smoothing.

    state_transition : {"return_to_baseline", "uniform",
                        "uniform_except_self", "custom"},
                       default="return_to_baseline"
        Transition rule governing the latent state sequence.

        - "return_to_baseline":
          State 0 transitions uniformly to any burst state,
          and all burst states transition back to state 0.

        - "uniform":
          Sample the next state uniformly from all states,
          including the current state.

        - "uniform_except_self":
          Sample the next state uniformly from all states
          except the current state.

        - "custom":
          Use the transition probabilities specified by
          `transition_matrix`.

    transition_matrix : array-like or None, default=None
        Custom state transition probability matrix.

        Must have shape (num_states, num_states), where each row contains
        the probabilities of transitioning from the current state to
        each possible next state.

        Used only when state_transition="custom".

    burst_amp_sampling : {"absolute_normal", "choice"},
                         default="absolute_normal"
        Method for sampling the amplitude multiplier of each burst.

        - "absolute_normal":
          Sample the absolute value of a normally distributed random
          variable with mean 1 and standard deviation `burst_amp_sigma`.

        - "choice":
          Randomly select an amplitude multiplier from
          `burst_amp_factors`.

        One multiplier is sampled independently for each burst.

    burst_amp_factors : array-like or None, default=None
        Discrete amplitude multipliers used when
        burst_amp_sampling="choice".

        For example, [1, 2] produces two possible amplitude multipliers
        with a relative amplitude ratio of 1:2.

        These factors control relative rather than absolute amplitudes.
        Final amplitudes may also depend on frequency-dependent scaling
        and subsequent SNR normalization.

        Ignored when burst_amp_sampling="absolute_normal".

    burst_amp_sigma : float, default=0.1
        Standard deviation of the normal distribution used to sample
        burst amplitude multipliers when
        burst_amp_sampling="absolute_normal".

        The multiplier is sampled as abs(N(1, burst_amp_sigma**2)).
        A single multiplier is sampled per burst, not per cycle.

    chi : float, default=0.15
        Exponent for optional frequency-dependent amplitude scaling.

        When enabled, burst amplitude is multiplied by f**(-chi).

    use_tukey : bool, default=True
        Whether to apply a Tukey window to each burst.

    tukey_alpha : float, default=0.25
        Shape parameter of the Tukey window.

    rise_ratio : float or array-like, default=0.3
        Rise-decay symmetry of asymmetric sine oscillations.

        Defined as the fraction of an oscillatory cycle spent rising
        from trough to peak:

            rdsym = rise_duration / (rise_duration + decay_duration)

        - 0.5: Symmetric oscillation.
        - < 0.5: Shorter rise and longer decay.
        - > 0.5: Longer rise and shorter decay.

        Values must be strictly between 0 and 1.

        - If scalar, use a fixed rise-decay ratio for all bursts.
        - If array-like, sample according to `rise_ratio_sampling`.

        One ratio is sampled per burst and remains constant
        across its oscillatory cycles.

        Used only when burst_type="asine".

    rise_ratio_sampling : {"uniform", "choice"}, default="uniform"
        Sampling method for rise_ratio when a sequence is provided.

        - "uniform":
          Sample continuously between two specified bounds.

        - "choice":
          Randomly select from the specified values.

        Ignored when rise_ratio is scalar.
        Used only when burst_type="asine".

    sawtooth_width : float or array-like, default=0.7
        Controls the asymmetry of the sawtooth waveform.

        - If scalar, use a fixed width for all bursts.
        - If array-like, sample according to `sawtooth_width_sampling`.

        Values must lie between 0 and 1 (inclusive).

        - 0.0: Descending sawtooth wave.
        - 0.5: Symmetric triangular wave.
        - 1.0: Ascending sawtooth wave.

        Used only when burst_type="sawtooth".

    sawtooth_smooth_fraction : float, default=0.06
        Controls Gaussian smoothing of the sawtooth waveform.

        Specifies the Gaussian standard deviation as a fraction
        of one oscillation cycle:

            sigma = sawtooth_smooth_fraction * fs / f

        Larger values produce smoother waveforms by attenuating
        higher-frequency components. Set to 0 to disable smoothing.

        Used only when burst_type="sawtooth".

    sawtooth_width_sampling : {"uniform", "choice"}, default="uniform"
        Sampling method for sawtooth_width when a sequence is provided.

        - "uniform":
          Sample continuously between two specified bounds.

        - "choice":
          Randomly select from the specified values.

        Ignored when sawtooth_width is scalar.
        Used only when burst_type="sawtooth".

    power_law_scale : bool, default=False
        Whether to apply frequency-dependent amplitude scaling
        proportional to f**(-chi).

        Scaling is applied only when multiple burst frequencies
        are specified.

    rng : np.random.Generator or None, default=None
        Random number generator used for stochastic sampling.

        If None, a new generator is created using np.random.default_rng().

    Returns
    -------
    signal : ndarray, shape (T,)
        Simulated signal containing oscillatory bursts and zero-valued
        baseline segments.

    states : ndarray, shape (T,)
        Integer-valued latent state sequence.

        - State 0: Baseline (noise) state.
        - States 1 through K: Oscillatory burst states.

    Notes
    -----
    Burst amplitudes are determined by sampled amplitude multipliers,
    with optional frequency-dependent scaling.

    If a common SNR normalization factor is subsequently applied to
    the generated signal, relative burst amplitude differences
    are preserved.

    Asymmetric sine bursts are generated using NeuroDSP's
    `sim_oscillation` function with `cycle="asine"`.

    NeuroDSP constructs asymmetric sine waves by varying the relative
    durations of their rising and decaying portions.

    The resulting nonsinusoidal oscillations contain harmonics,
    even though harmonics are not explicitly summed.

    NeuroDSP discretizes individual cycles, so the realized
    oscillation frequency can differ slightly from the requested
    frequency when fs / f is not an integer.

    References
    ----------
    Cole, S., Donoghue, T., Gao, R., & Voytek, B. (2019).
    NeuroDSP: A package for neural digital signal processing.
    Journal of Open Source Software, 4(36), 1272.
    https://doi.org/10.21105/joss.01272
    """

    # ============================================================
    # Initialize random number generator
    # ============================================================

    if rng is None:
        rng = np.random.default_rng()

    # ============================================================
    # Validate parameters
    # ============================================================

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("`fs` must be finite and positive.")

    freqs_array = np.atleast_1d(
        np.asarray(f, dtype=float)
    )

    if (
        freqs_array.ndim != 1
        or freqs_array.size == 0
        or not np.all(np.isfinite(freqs_array))
        or np.any(freqs_array <= 0)
    ):
        raise ValueError(
            "`f` must contain finite, positive frequencies."
        )

    if burst_type not in ("sine", "asine", "sawtooth"):
        raise ValueError(
            "`burst_type` must be 'sine', 'asine', or 'sawtooth'."
        )

    if burst_amp_sampling not in ("absolute_normal", "choice"):
        raise ValueError(
            "`burst_amp_sampling` must be "
            "'absolute_normal' or 'choice'."
        )

    if burst_amp_sampling == "choice":

        if burst_amp_factors is None:
            raise ValueError(
                "`burst_amp_factors` must be provided "
                "when burst_amp_sampling='choice'."
            )

        burst_amp_factors = np.atleast_1d(
            np.asarray(burst_amp_factors, dtype=float)
        )

        if (
            burst_amp_factors.size == 0
            or not np.all(np.isfinite(burst_amp_factors))
            or np.any(burst_amp_factors <= 0)
        ):
            raise ValueError(
                "`burst_amp_factors` must contain finite, "
                "positive values."
            )

    elif burst_amp_sigma < 0 or not np.isfinite(burst_amp_sigma):
        raise ValueError(
            "`burst_amp_sigma` must be finite and non-negative."
        )

    # Validate asymmetric sine parameters.

    if burst_type == "asine":

        rise_values = np.atleast_1d(
            np.asarray(rise_ratio, dtype=float)
        )

        if (
            rise_values.ndim != 1
            or rise_values.size == 0
            or not np.all(np.isfinite(rise_values))
            or np.any(rise_values <= 0)
            or np.any(rise_values >= 1)
        ):
            raise ValueError(
                "`rise_ratio` must contain finite values "
                "strictly between 0 and 1."
            )

        if rise_values.size > 1:

            if rise_ratio_sampling not in ("uniform", "choice"):
                raise ValueError(
                    "`rise_ratio_sampling` must be "
                    "'uniform' or 'choice'."
                )

            if rise_ratio_sampling == "uniform":

                if rise_values.size != 2:
                    raise ValueError(
                        "Uniform sampling requires a length-2 "
                        "`rise_ratio` range."
                    )

                if rise_values[0] > rise_values[1]:
                    raise ValueError(
                        "`rise_ratio` bounds must be "
                        "in ascending order."
                    )

    # Validate sawtooth parameters.

    if burst_type == "sawtooth":

        width_values = np.atleast_1d(
            np.asarray(sawtooth_width, dtype=float)
        )

        if (
            width_values.ndim != 1
            or width_values.size == 0
            or not np.all(np.isfinite(width_values))
            or np.any(width_values < 0)
            or np.any(width_values > 1)
        ):
            raise ValueError(
                "`sawtooth_width` must contain finite values "
                "between 0 and 1."
            )

        if width_values.size > 1:

            if sawtooth_width_sampling not in ("uniform", "choice"):
                raise ValueError(
                    "`sawtooth_width_sampling` must be "
                    "'uniform' or 'choice'."
                )

            if sawtooth_width_sampling == "uniform":

                if width_values.size != 2:
                    raise ValueError(
                        "Uniform sampling requires a length-2 "
                        "`sawtooth_width` range."
                    )

                if width_values[0] > width_values[1]:
                    raise ValueError(
                        "`sawtooth_width` bounds must be "
                        "in ascending order."
                    )

        if (
            not np.isfinite(sawtooth_smooth_fraction)
            or sawtooth_smooth_fraction < 0
        ):
            raise ValueError(
                "`sawtooth_smooth_fraction` must be "
                "finite and non-negative."
            )

    # Initialize signal and states

    use_scalar_freq = freqs_array.size == 1

    num_states = freqs_array.size + 1  # State 0 = baseline

    init_p = np.ones(num_states, dtype=float) / num_states

    signal = np.zeros(len(time_vec), dtype=float)
    states = np.zeros(len(time_vec), dtype=int)

    trans_mat = make_transition_matrix(
        num_states=num_states,
        state_transition=state_transition,
        trans_mat=transition_matrix,
    )

    start_idx = 0
    last_state = None

    # Generate signal

    while start_idx < len(time_vec):

        # Sample current state

        if last_state is None:

            current_state = rng.choice(
                num_states,
                p=init_p,
            )

        else:

            current_state = rng.choice(
                num_states,
                p=trans_mat[last_state],
            )

        last_state = current_state

        # Baseline segment

        if current_state == 0:

            curr_noise_duration = _get_duration(
                noise_duration,
                rng,
            )

            n_noise_samples = int(
                round(curr_noise_duration * fs)
            )

            if n_noise_samples <= 0:
                raise ValueError(
                    "`noise_duration` must produce "
                    "at least one sample."
                )

            end_idx = min(
                start_idx + n_noise_samples,
                len(time_vec),
            )

            states[start_idx:end_idx] = 0

        # Oscillatory burst segment

        else:

            # Burst frequency

            freq_burst = (
                freqs_array[0]
                if use_scalar_freq
                else freqs_array[current_state - 1]
            )

            # Number of cycles

            if isinstance(
                burst_cycles,
                (list, tuple, np.ndarray),
            ):

                if len(burst_cycles) != 2:
                    raise ValueError(
                        "`burst_cycles` must be a scalar "
                        "or a two-element sequence."
                    )

                num_cycles = int(
                    rng.integers(
                        burst_cycles[0],
                        burst_cycles[1] + 1,
                    )
                )

            elif np.isscalar(burst_cycles):

                num_cycles = int(burst_cycles)

            else:

                raise ValueError(
                    "`burst_cycles` must be a scalar "
                    "or a two-element sequence."
                )

            if num_cycles <= 0:
                raise ValueError(
                    "`burst_cycles` must be positive."
                )

            # Burst duration

            total_samples = int(
                round(num_cycles * fs / freq_burst)
            )

            if total_samples <= 0:
                raise ValueError(
                    "Invalid total_samples. "
                    "Check `fs`, `f`, and `burst_cycles`."
                )

            aligned_time = np.arange(total_samples) / fs

            # Generate oscillatory carrier

            # 1. Sinusoidal waveform

            if burst_type == "sine":

                carrier = np.sin(
                    2 * np.pi * freq_burst * aligned_time
                )

            # 2. Asymmetric sine waveform (NeuroDSP)

            elif burst_type == "asine":

                # Sample one rise-decay ratio per burst.

                if rise_values.size == 1:

                    rdsym = float(rise_values[0])

                elif rise_ratio_sampling == "uniform":

                    rdsym = rng.uniform(
                        rise_values[0],
                        rise_values[1],
                    )

                else:  # choice

                    rdsym = rng.choice(
                        rise_values
                    )

                # NeuroDSP generates asymmetric sine waves by
                # modifying the relative durations of rising
                # and decaying portions of the oscillatory cycle.
                #
                # rdsym = 0.5: symmetric oscillation
                # rdsym < 0.5: faster rise, slower decay
                # rdsym > 0.5: slower rise, faster decay
                #
                # Reference:
                # Cole et al. (2019), JOSS, 4(36), 1272.
                # https://doi.org/10.21105/joss.01272

                carrier = sim_oscillation(
                    n_seconds=total_samples / fs,
                    fs=fs,
                    freq=freq_burst,
                    cycle="asine",
                    rdsym=rdsym,
                )

                # NeuroDSP constructs oscillations by repeating
                # discretized cycles. At some frequencies, the
                # resulting array may be shorter than requested.
                #
                # Repeat or truncate the periodic waveform to
                # ensure the exact requested number of samples.

                if len(carrier) == 0:
                    raise ValueError(
                        "NeuroDSP returned an empty carrier. "
                        "Check `fs` and `freq_burst`."
                    )

                if len(carrier) != total_samples:

                    carrier = np.resize(
                        carrier,
                        total_samples,
                    )

            # 3. Sawtooth waveform

            elif burst_type == "sawtooth":

                # Sample one width per burst.

                if width_values.size == 1:

                    width = float(width_values[0])

                elif sawtooth_width_sampling == "uniform":

                    width = rng.uniform(
                        width_values[0],
                        width_values[1],
                    )

                else:  # choice

                    width = rng.choice(
                        width_values
                    )

                phase = (
                    2 * np.pi * freq_burst * aligned_time
                )

                carrier = sawtooth(
                    phase + np.pi * width,
                    width=width,
                )

                # Apply Gaussian smoothing.

                if sawtooth_smooth_fraction > 0:

                    sigma = (
                        sawtooth_smooth_fraction
                        * fs
                        / freq_burst
                    )

                    carrier = gaussian_filter1d(
                        carrier,
                        sigma=sigma,
                        mode="nearest",
                    )

            # Burst envelope

            if use_tukey:

                envelope = tukey(
                    total_samples,
                    alpha=tukey_alpha,
                )

            else:

                envelope = np.ones(total_samples)

            # Burst amplitude

            if burst_amp_sampling == "absolute_normal":

                burst_amplitude = np.abs(
                    rng.normal(
                        loc=1.0,
                        scale=burst_amp_sigma,
                    )
                )

            else:  # choice

                burst_amplitude = rng.choice(
                    burst_amp_factors
                )

            # Optional frequency-dependent scaling.

            if (not use_scalar_freq) and power_law_scale:

                burst_amplitude *= (
                    1.0 / (freq_burst ** chi)
                )

            # Construct burst
            burst_signal = (
                burst_amplitude
                * carrier
                * envelope
            )

            # Truncate if the burst exceeds signal boundaries.

            end_idx = min(
                start_idx + total_samples,
                len(time_vec),
            )

            burst_signal = burst_signal[
                :end_idx - start_idx
            ]

            signal[start_idx:end_idx] = burst_signal
            states[start_idx:end_idx] = current_state

        start_idx = end_idx

    return signal, states


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


import numpy as np
from neurodsp.filt import filter_signal


def _add_noise(
    bursts,
    states,
    noise,
    snr_db=None,
    use_filter=True,
    fs=None,
    highpass_f=0.5,
    mode="state_snr",
    noise_scale=1.0,
):
    """
    Mix a clean burst signal with noise using different scaling modes.

    Parameters
    ----------
    bursts : 1-D array
        Clean burst signal containing zeros during noise gaps.
    states : 1-D int array
        State labels (0 = noise gaps, >0 = recurring burst states).
    noise : 1-D array
        Noise signal with the same length as bursts.
    snr_db : float, optional
        Target SNR in dB for 'state_snr' and 'global_snr'.
        SNR is defined relative to the variance of filtered noise
        when use_filter=True.
    use_filter : bool, default=True
        Whether to high-pass filter noise for SNR calculation.
        The original, unfiltered noise is always added to the signal.
    fs : float, optional
        Sampling frequency in Hz.
    highpass_f : float, default=0.5
        High-pass cutoff frequency in Hz.
    mode : {'state_snr', 'global_snr', 'noise_scale'}
        Method used to mix bursts and noise.

        'state_snr':
            Scale bursts independently for each state to achieve
            the specified state-level SNR. Within-state amplitude
            variability is preserved.

        'global_snr':
            Scale noise to achieve the specified global SNR
            across all burst-active samples. Burst amplitudes
            remain unchanged.

        'noise_scale':
            Multiply the original noise by a custom amplitude
            scaling factor. Burst amplitudes remain unchanged.

    noise_scale : float, default=1.0
        Noise amplitude scaling factor when mode='noise_scale'.

    Returns
    -------
    noisy : 1-D array
        Burst signal mixed with noise.
    scaled_component : 1-D array
        Scaled bursts when mode='state_snr'.
        Scaled noise when mode='global_snr' or 'noise_scale'.
    """

    if bursts.shape != noise.shape or bursts.shape != states.shape:
        raise ValueError(
            "bursts, noise, and states must have identical shapes"
        )

    if use_filter and (fs is None or highpass_f is None):
        raise ValueError(
            "`fs` and `highpass_f` must be provided when use_filter=True"
        )

    valid_modes = ("state_snr", "global_snr", "noise_scale")

    if mode not in valid_modes:
        raise ValueError(
            f"Unknown mode: {mode}. Choose from {valid_modes}"
        )

    if mode in ("state_snr", "global_snr"):
        if snr_db is None or not np.isfinite(snr_db):
            raise ValueError(
                "`snr_db` must be finite for SNR-based modes."
            )

    bursts_copy = np.asarray(bursts, dtype=float).copy()
    noise_copy = np.asarray(noise, dtype=float).copy()

    # Filter noise only for SNR calibration.
    # The original noise is retained for mixing.
    if mode in ("state_snr", "global_snr"):

        if use_filter:
            filtered_noise = filter_signal(
                noise_copy,
                fs,
                "highpass",
                f_range=(highpass_f, None),
                remove_edges=False,
            )
        else:
            filtered_noise = noise_copy

        noise_power = np.var(filtered_noise)

        if noise_power <= 0:
            raise ValueError(
                "Reference noise power must be positive."
            )

        snr_linear = 10 ** (snr_db / 10)

    if mode == "state_snr":

        # Normalize the mean power of each state.
        # Within-state amplitude variability is preserved.
        desired_burst_power = snr_linear * noise_power

        for s in np.unique(states):

            if s == 0:
                continue

            mask = states == s

            burst_power = np.mean(
                bursts_copy[mask] ** 2
            )

            if burst_power <= 0:
                raise ValueError(
                    f"State {s} has zero burst power."
                )

            scaling_factor = np.sqrt(
                desired_burst_power / burst_power
            )

            bursts_copy[mask] *= scaling_factor

        scaled_component = bursts_copy

    elif mode == "global_snr":

        # Scale noise relative to global burst power.
        # Original burst amplitudes are preserved.
        burst_mask = states != 0

        if not np.any(burst_mask):
            raise ValueError(
                "No burst-active samples found."
            )

        burst_power = np.mean(
            bursts_copy[burst_mask] ** 2
        )

        if burst_power <= 0:
            raise ValueError(
                "Global burst power must be positive."
            )

        desired_noise_power = burst_power / snr_linear

        scaling_factor = np.sqrt(
            desired_noise_power / noise_power
        )

        noise_copy *= scaling_factor

        scaled_component = noise_copy

    elif mode == "noise_scale":

        # Apply a custom noise amplitude scaling factor.
        # Original burst amplitudes are preserved.
        if not np.isfinite(noise_scale) or noise_scale < 0:
            raise ValueError(
                "`noise_scale` must be finite and nonnegative."
            )

        noise_copy *= noise_scale

        scaled_component = noise_copy

    # Add the original (unfiltered) noise,
    # optionally scaled according to the selected mode.
    noisy = bursts_copy + noise_copy

    return noisy, scaled_component

def simulate_bursty_signal(
    time_vec,
    fs,
    freq,
    burst_cycles_param,
    noise_duration_param,
    state_transition="uniform_except_self",
    transition_matrix=None,
    burst_type="sine",
    noise_addition_mode="state_snr",
    snr_db=2,
    use_filter=True,
    highpass_f=0.5,
    noise_scale=1.0,
    burst_amp_sampling="absolute_normal",
    burst_amp_factors=None,
    burst_amp_sigma=0.1,
    beta=1,
    chi=0.15,
    power_law_scale=False,
    rise_ratio=0.3,
    rise_ratio_sampling="uniform",
    sawtooth_width=0.7,
    sawtooth_smooth_fraction=0.08,
    sawtooth_width_sampling="uniform",
    use_tukey=True,
    tukey_alpha=0.25,
    rng=None,
    custom_noise=None,
):
    """
    Simulate a bursty signal with added noise.

    Parameters
    ----------
    noise_addition_mode : {'state_snr', 'global_snr', 'noise_scale', 'unscaled'}
        Method used to combine bursts and noise.

        'state_snr':
            Scale bursts independently for each state to achieve
            the specified state-wise SNR. Relative amplitudes
            within each state are preserved.

        'global_snr':
            Scale noise to achieve the specified global SNR
            across all burst-active samples. Burst amplitudes
            remain unchanged.

        'noise_scale':
            Multiply noise by the specified noise_scale factor.
            Burst amplitudes remain unchanged.

        'unscaled':
            Add noise directly without scaling bursts or noise.
            The snr_db, noise_scale, and use_filter parameters
            are ignored.

    snr_db : float, default=2
        Target SNR in dB for 'state_snr' and 'global_snr'.
        When use_filter=True, SNR is calibrated relative to the
        variance of high-pass-filtered noise, although the
        original, unfiltered noise is added to the signal.

    use_filter : bool, default=True
        Whether to high-pass filter noise for SNR calibration.
        Filtering does not modify the noise added to the signal.

    highpass_f : float, default=0.5
        High-pass cutoff frequency in Hz for SNR calibration.

    noise_scale : float, default=1.0
        Noise amplitude scaling factor when
        noise_addition_mode='noise_scale'.

    custom_noise : 1-D array, optional
        Custom noise signal with the same length as time_vec.
        If None, colored noise is generated internally.
        The supplied noise is not modified in place.

    Returns
    -------
    dict
        Dictionary containing:

        'signal':
            Simulated signal (bursts + noise).
        'states':
            Ground-truth state sequence.
        'bursts':
            Burst component after any scaling.
        'noise':
            Noise component after any scaling.

        The returned components satisfy:
            signal = bursts + noise
    """

    valid_modes = (
        "state_snr",
        "global_snr",
        "noise_scale",
        "unscaled",
    )

    if noise_addition_mode not in valid_modes:
        raise ValueError(
            f"Invalid noise_addition_mode: {noise_addition_mode}. "
            f"Choose from {valid_modes}."
        )

    if rng is None:
        rng = np.random.default_rng()

    # Generate bursts.
    bursts, states = _simulate_bursts(
        time_vec,
        fs,
        freq,
        burst_cycles_param,
        noise_duration_param,
        state_transition=state_transition,
        transition_matrix=transition_matrix,
        burst_type=burst_type,
        burst_amp_sampling=burst_amp_sampling,
        burst_amp_factors=burst_amp_factors,
        burst_amp_sigma=burst_amp_sigma,
        chi=chi,
        power_law_scale=power_law_scale,
        rise_ratio=rise_ratio,
        rise_ratio_sampling=rise_ratio_sampling,
        sawtooth_width=sawtooth_width,
        sawtooth_smooth_fraction=sawtooth_smooth_fraction,
        sawtooth_width_sampling=sawtooth_width_sampling,
        use_tukey=use_tukey,
        tukey_alpha=tukey_alpha,
        rng=rng,
    )

    # Generate colored noise unless custom noise is provided.
    if custom_noise is None:
        noise = _generate_colored_noise(
            len(time_vec),
            fs,
            beta,
            rng=rng,
        )

    else:
        noise = np.asarray(
            custom_noise,
            dtype=float,
        ).copy()

        if noise.shape != (len(time_vec),):
            raise ValueError(
                "`custom_noise` must be a 1-D array "
                "with the same length as `time_vec`."
            )

    # Add noise without scaling either component.
    if noise_addition_mode == "unscaled":

        signal = bursts + noise

    else:

        signal, scaled_component = _add_noise(
            bursts,
            states,
            noise,
            snr_db=snr_db,
            use_filter=use_filter,
            fs=fs,
            highpass_f=highpass_f,
            mode=noise_addition_mode,
            noise_scale=noise_scale,
        )

        if noise_addition_mode == "state_snr":
            bursts = scaled_component

        else:
            noise = scaled_component

    return {
        "signal": signal,
        "states": states,
        "bursts": bursts,
        "noise": noise,
    }