"""
The following burst simulator is inspired from:
- "Quinn2019_BurstHMM/hmm_util_get_simulation.m"
- "Cho2022_BurstDetection/utils/util_data/generates_simulaiton.m"
- Cole, S., Donoghue, T., Gao, R., & Voytek, B. (2019). NeuroDSP: A package for
neural digital signal processing. Journal of Open Source Software, 4(36), 1272.
DOI: 10.21105/joss.01272
"""


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


from neurodsp.sim import sim_oscillation
from scipy.ndimage import gaussian_filter1d
from scipy.signal import sawtooth
from scipy.signal.windows import tukey


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
    power_law_scale=False,
    rng=None,
    sawtooth_smooth_fraction=0.0,
):
    """
    Generate oscillatory bursts interspersed with zero-valued baseline.

    Parameters
    ----------
    time_vec : array-like
        Determines output length. Assumed uniformly sampled at `fs`.
    fs : float
        Sampling frequency in Hz.
    f : float or array-like
        Burst frequencies; one distinct state per frequency.
    burst_cycles : int or (int, int)
        Fixed cycle count or inclusive lower/upper bounds.
    noise_duration : float or (float, float)
        Baseline duration or uniform range in seconds.
    burst_type : {"sine", "asine", "sawtooth"}
        Type of oscillatory carrier.
    state_transition : str
        Transition scheme passed to `make_transition_matrix`.
    transition_matrix : array-like or None
        Optional custom transition probabilities.
    burst_amp_sampling : {"absolute_normal", "choice"}
        Amplitude multiplier sampling method.
    burst_amp_factors : array-like or None
        Amplitude choices if `burst_amp_sampling="choice"`.
    burst_amp_sigma : float
        Standard deviation for abs(N(1, burst_amp_sigma**2)).
    chi : float
        Power-law amplitude exponent.
    use_tukey : bool
        Whether to apply a Tukey burst envelope.
    tukey_alpha : float
        Tukey window shape parameter in [0, 1].
    rise_ratio : float or array-like
        Fraction of a cycle spent rising from trough to peak.
        Shared by "asine" and "sawtooth"; ignored for "sine".
        Values must lie strictly between 0 and 1.
    rise_ratio_sampling : {"uniform", "choice"}
        For multiple `rise_ratio` values: "uniform" requires
        two ordered bounds, while "choice" accepts candidate values.
        Sampled once per burst.
    power_law_scale : bool
        Apply f**(-chi) to burst amplitude when there are
        multiple burst-frequency states.
    rng : np.random.Generator or None
        Random number generator.
    sawtooth_smooth_fraction : float
        Gaussian smoothing sigma as a fraction of cycle length;
        0 disables smoothing. Only used for "sawtooth".

    Returns
    -------
    signal : ndarray, shape (T,)
        Burst signal with zero-valued baseline segments.
    states : ndarray, shape (T,)
        Latent states: 0 for baseline, 1..K for bursts.

    Notes
    -----

    NeuroDSP's "asine" uses discretized repeated cycles, so its
    realized frequency may differ from `f` if `fs / f` is noninteger.

    References
    ----------
    Cole et al. (2019). NeuroDSP: A package for neural digital
    signal processing. JOSS, 4(36), 1272.
    https://doi.org/10.21105/joss.01272
    """
    if rng is None:
        rng = np.random.default_rng()

    # ============================================================
    # Validate parameters
    # ============================================================

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("`fs` must be finite and positive.")

    freqs_array = np.atleast_1d(np.asarray(f, dtype=float))

    if burst_type not in ("sine", "asine", "sawtooth"):
        raise ValueError("Invalid `burst_type`.")
    if burst_amp_sampling not in ("absolute_normal", "choice"):
        raise ValueError("Invalid `burst_amp_sampling`.")
    if not np.isfinite(chi):
        raise ValueError("`chi` must be finite.")
    if not np.isfinite(tukey_alpha) or not 0 <= tukey_alpha <= 1:
        raise ValueError("`tukey_alpha` must be in [0, 1].")

    # Validate amplitude parameters.

    if burst_amp_sampling == "choice":
        if burst_amp_factors is None:
            raise ValueError("`burst_amp_factors` is required for 'choice'.")
        burst_amp_factors = np.atleast_1d(
            np.asarray(burst_amp_factors, dtype=float)
        )
        if (
            burst_amp_factors.ndim != 1
            or burst_amp_factors.size == 0
            or not np.all(np.isfinite(burst_amp_factors))
            or np.any(burst_amp_factors <= 0)
        ):
            raise ValueError("`burst_amp_factors` must be positive and finite.")
    elif not np.isfinite(burst_amp_sigma) or burst_amp_sigma < 0:
        raise ValueError("`burst_amp_sigma` must be finite and nonnegative.")

    # Validate burst cycle counts.

    cycle_values = np.atleast_1d(np.asarray(burst_cycles, dtype=float))
    if (
        cycle_values.ndim != 1
        or cycle_values.size not in (1, 2)
        or not np.all(np.isfinite(cycle_values))
        or np.any(cycle_values <= 0)
        or not np.all(cycle_values == np.floor(cycle_values))
        or (cycle_values.size == 2 and cycle_values[0] > cycle_values[1])
    ):
        raise ValueError(
            "`burst_cycles` must be a positive integer or ordered pair."
        )
    cycle_values = cycle_values.astype(int)

    # Validate baseline durations.

    noise_values = np.atleast_1d(np.asarray(noise_duration, dtype=float))
    if (
        noise_values.ndim != 1
        or noise_values.size not in (1, 2)
        or not np.all(np.isfinite(noise_values))
        or np.any(noise_values <= 0)
        or (noise_values.size == 2 and noise_values[0] > noise_values[1])
    ):
        raise ValueError(
            "`noise_duration` must be positive or ordered bounds."
        )

    # ============================================================
    # Validate shared rise-decay parameters
    # ============================================================

    if burst_type in ("asine", "sawtooth"):
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
                "`rise_ratio` values must be strictly in (0, 1)."
            )

        if rise_values.size > 1:
            if rise_ratio_sampling not in ("uniform", "choice"):
                raise ValueError("Invalid `rise_ratio_sampling`.")

            if rise_ratio_sampling == "uniform" and (
                rise_values.size != 2 or rise_values[0] > rise_values[1]
            ):
                raise ValueError(
                    "Uniform sampling requires ordered 2-item bounds."
                )

    if burst_type == "sawtooth" and (
        not np.isfinite(sawtooth_smooth_fraction)
        or sawtooth_smooth_fraction < 0
    ):
        raise ValueError(
            "`sawtooth_smooth_fraction` must be nonnegative."
        )

    # ============================================================
    # Initialize signal and states
    # ============================================================

    num_states = freqs_array.size + 1
    n_samples = len(time_vec)

    signal = np.zeros(n_samples, dtype=float)
    states = np.zeros(n_samples, dtype=int)

    init_p = np.full(num_states, 1 / num_states)

    trans_mat = make_transition_matrix(
        num_states=num_states,
        state_transition=state_transition,
        trans_mat=transition_matrix,
    )

    start_idx = 0
    last_state = None

    # ============================================================
    # Generate signal
    # ============================================================

    while start_idx < n_samples:

        # Sample the next state.

        if last_state is None:
            current_state = rng.choice(num_states, p=init_p)
        else:
            current_state = rng.choice(
                num_states, p=trans_mat[last_state]
            )

        last_state = current_state

        # --------------------------------------------------------
        # Baseline segment
        # --------------------------------------------------------

        if current_state == 0:

            curr_noise_duration = _get_duration(noise_duration, rng)

            n_noise_samples = int(round(curr_noise_duration * fs))

            if n_noise_samples <= 0:
                raise ValueError(
                    "`noise_duration` must produce >= 1 sample."
                )

            end_idx = min(start_idx + n_noise_samples, n_samples)

        # --------------------------------------------------------
        # Oscillatory burst segment
        # --------------------------------------------------------

        else:

            # Burst frequency.

            freq_burst = freqs_array[current_state - 1]

            # Number of cycles.

            if cycle_values.size == 1:
                num_cycles = int(cycle_values[0])
            else:
                num_cycles = int(
                    rng.integers(cycle_values[0], cycle_values[1] + 1)
                )

            # Burst duration.

            total_samples = int(round(num_cycles * fs / freq_burst))

            if total_samples <= 0:
                raise ValueError(
                    "Burst duration must be >= 1 sample."
                )

            aligned_time = np.arange(total_samples) / fs

            # ====================================================
            # Generate oscillatory carrier
            # ====================================================

            # 1. Standard sine wave.

            if burst_type == "sine":

                carrier = np.sin(
                    2 * np.pi * freq_burst * aligned_time
                )

            # 2. Asymmetric sine or sawtooth wave.

            else:

                # Sample one rise-decay ratio per burst.
                # The same parameter controls both waveform types.

                if rise_values.size == 1:

                    rdsym = float(rise_values[0])

                elif rise_ratio_sampling == "uniform":

                    rdsym = rng.uniform(
                        rise_values[0],
                        rise_values[1],
                    )

                else:  # choice

                    rdsym = float(rng.choice(rise_values))

                # -----------------------------------------------
                # Asymmetric sine (NeuroDSP)
                # -----------------------------------------------

                if burst_type == "asine":

                    carrier = np.asarray(
                        sim_oscillation(
                            n_seconds=total_samples / fs,
                            fs=fs,
                            freq=freq_burst,
                            cycle="asine",
                            rdsym=rdsym,
                        ),
                        dtype=float,
                    )

                    if carrier.size == 0:
                        raise ValueError(
                            "NeuroDSP returned an empty carrier."
                        )

                    # Extend a tiled periodic cycle when the library
                    # returns fewer samples than requested.

                    if carrier.size < total_samples:
                        n_repeats = int(
                            np.ceil(total_samples / carrier.size)
                        )
                        carrier = np.tile(carrier, n_repeats)

                    carrier = carrier[:total_samples]

                # -----------------------------------------------
                # Sawtooth wave (SciPy)
                # -----------------------------------------------

                else:

                    phase = (
                        2 * np.pi * freq_burst * aligned_time
                    )

                    # width = fraction of cycle spent rising.
                    # Phase shift starts at the rising zero crossing.

                    carrier = sawtooth(
                        phase + np.pi * rdsym,
                        width=rdsym,
                    )

                    # Optional Gaussian smoothing.

                    if sawtooth_smooth_fraction > 0:

                        sigma_samples = (
                            sawtooth_smooth_fraction
                            * fs / freq_burst
                        )

                        carrier = gaussian_filter1d(
                            carrier,
                            sigma=sigma_samples,
                            mode="wrap",
                        )

            # ====================================================
            # Burst envelope
            # ====================================================

            envelope = (
                tukey(total_samples, alpha=tukey_alpha)
                if use_tukey
                else np.ones(total_samples)
            )

            # ====================================================
            # Burst amplitude
            # ====================================================

            if burst_amp_sampling == "absolute_normal":

                burst_amplitude = abs(
                    rng.normal(
                        loc=1.0,
                        scale=burst_amp_sigma,
                    )
                )

            else:  # choice

                burst_amplitude = float(
                    rng.choice(burst_amp_factors)
                )

            # Optional frequency-dependent scaling.

            if freqs_array.size > 1 and power_law_scale:

                burst_amplitude *= freq_burst ** (-chi)

            # ====================================================
            # Construct burst
            # ====================================================

            burst_signal = (
                burst_amplitude * carrier * envelope
            )

            # Truncate at the signal boundary.

            end_idx = min(start_idx + total_samples, n_samples)

            signal[start_idx:end_idx] = burst_signal[
                :end_idx - start_idx
            ]

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