from scipy.signal import welch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from nds_toolbox.sim.bursts.simulator import simulate_bursty_signal
from matplotlib.colors import Normalize

# %%

simulation_condition = "prior_sensitivity"
figure_dir = f"../figures/{simulation_condition}"
data_dir = "../data/simulations"
data_file = f"../data/simulations/{simulation_condition}_data.npz"

os.makedirs(figure_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# global settings (used across all the simulations)

# Set seeds for reproducibility.
seed_base = 2026

# Simulation parameters.
n_samples = 10

fs = 250  # Sampling frequency in Hz.
n_seconds = 60 * 3  # Total duration in seconds.
time_vec = np.arange(int(fs * n_seconds)) / fs

burst_amp_sigma = 0.1

tukey_alpha = 0.25
state_transition = 'return_to_baseline'

beta = 1  # pink noise

# For burst segments, specify duration as the number of cycles.
burst_cycles = [3, 10]

# For noise segments, specify duration in seconds.
noise_duration = [0.2, 2]

freq = 20

snr_db = 2
burst_type = "sine"

###################################################################

# Initialize variables
states_sample = np.zeros((n_samples, len(time_vec)))
bursts_sample = np.zeros((n_samples, len(time_vec)))
noise_sample = np.zeros((n_samples, len(time_vec)))
signal_sample = np.zeros((n_samples, len(time_vec)))

# Loop over the number of samples.
for sample_id in range(n_samples):
    # One seed per sample
    sample_seed = seed_base + sample_id

    # Reset RNG with the same sample seed
    rng = np.random.default_rng(sample_seed)

    # Simulate a signal with sine bursts.

    signal_dict = simulate_bursty_signal(time_vec, fs, freq, burst_cycles, noise_duration,
                                         state_transition=state_transition, burst_type=burst_type, use_filter=True,
                                         snr_db=snr_db, burst_amp_sigma=burst_amp_sigma, beta=beta,
                                         tukey_alpha=tukey_alpha, rng=rng)

    states_sample[sample_id, :] = signal_dict["states"]
    bursts_sample[sample_id, :] = signal_dict["bursts"]
    noise_sample[sample_id, :] = signal_dict["noise"]
    signal_sample[sample_id, :] = signal_dict["signal"]

# %%
print("Signal shape", signal_sample.shape, "[samples, data points]")

### save the data

data_file = f"{data_dir}/{simulation_condition}_data.npz"
np.savez_compressed(data_file,
                    signal_sample=signal_sample,
                    states_sample=states_sample,
                    bursts_sample=bursts_sample,
                    noise_sample=noise_sample)

print(f"Data saved as {data_file}")
