from scipy.signal import welch
import numpy as np
import os
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from nds_toolbox.sim.bursts.simulator import simulate_bursty_signal
from matplotlib.colors import Normalize

# ============================================================
# Directories
# ============================================================

simulation_condition = "signal_duration"

figure_dir = f"../figures/{simulation_condition}"
data_dir = "../data/simulations"

os.makedirs(figure_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# ============================================================
# Global simulation parameters
# ============================================================

seed_base = 2026
n_samples = 10

fs = 100

freq = [20]
snr_db = 2

burst_type = "sine"
burst_amp_sigma = 0.1

tukey_alpha = 0.25
state_transition = "return_to_baseline"

beta = 1  # Pink noise

# Burst duration in cycles
burst_cycles = [3, 10]

# Noise duration in seconds
noise_duration = [0.2, 2]

# Experimental condition: signal duration
n_seconds_vec = np.array([
    10,  # 10 sec
    30,  # 30 sec
    60,  # 1 min
    180,  # 3 min
    600,  # 10 min
    1800,  # 30 min
    5400,  # 1.5 hours
    18000,  # 5 hours
])
n_conditions = len(n_seconds_vec)

max_seconds = int(np.max(n_seconds_vec))
max_n_times = int(fs * max_seconds)

time_vec = np.arange(max_n_times) / fs

print("Maximum signal duration:", max_seconds, "seconds")
print("Maximum number of samples:", max_n_times)

# Initialize PSD storage

# Fixed Welch window across all duration conditions
nperseg = fs * 2

n_freqs = nperseg // 2 + 1

psds = np.zeros(
    (n_samples, n_conditions, n_freqs),
    dtype=np.float64,
)

# Store example traces from the first sample only
example_traces = {}

example_durations = [
    int(n_seconds_vec[0]),
    int(n_seconds_vec[2]),
    int(n_seconds_vec[-1]),
]

plot_seconds = 10
plot_n_times = int(plot_seconds * fs)

# ============================================================
# Simulate signals
# ============================================================

for sample_id in tqdm(
        range(n_samples),
        desc="Simulating",
):
    sample_seed = seed_base + sample_id

    rng = np.random.default_rng(sample_seed)

    # --------------------------------------------------------
    # Generate the longest signal once per sample
    # --------------------------------------------------------

    signal_dict = simulate_bursty_signal(
        time_vec,
        fs,
        freq,
        burst_cycles,
        noise_duration,
        state_transition=state_transition,
        burst_type=burst_type,
        use_filter=True,
        snr_db=snr_db,
        burst_amp_sigma=burst_amp_sigma,
        beta=beta,
        tukey_alpha=tukey_alpha,
        rng=rng,
    )

    # --------------------------------------------------------
    # Save the longest signal
    # --------------------------------------------------------

    data_file = os.path.join(
        data_dir,
        f"{simulation_condition}_sample_{sample_id}.npz",
    )

    np.savez_compressed(
        data_file,
        signal_sample=np.asarray(
            signal_dict["signal"],
            dtype=np.float32,
        ),
        states_sample=np.asarray(
            signal_dict["states"],
            dtype=np.int16,
        ),
        bursts_sample=np.asarray(
            signal_dict["bursts"],
            dtype=np.float32,
        ),
        noise_sample=np.asarray(
            signal_dict["noise"],
            dtype=np.float32,
        ),
        fs=fs,
        freq=freq,
        snr_db=snr_db,
        burst_amp_sigma=burst_amp_sigma,
        n_seconds_vec=n_seconds_vec,
        seed=sample_seed,
    )
