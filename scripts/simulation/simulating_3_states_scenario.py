from scipy.signal import welch
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from nds_toolbox.sim.bursts.simulator import simulate_bursty_signal
from matplotlib.colors import Normalize




#%%


FIG_WIDTH = 7.5
MY_FONT_SIZE = 9


plt.rcParams.update({
    # --- Fonts ---
    "font.family": "Arial",
    "font.size": MY_FONT_SIZE,

    # --- Legend ---
    "legend.fontsize": MY_FONT_SIZE,
    "legend.frameon": False,

    # --- Titles ---
    "figure.titlesize": MY_FONT_SIZE,
    "axes.titleweight": "normal",
    "axes.titlesize": MY_FONT_SIZE,

    # --- Axes ---
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "axes.unicode_minus": True,
    "axes.labelsize": MY_FONT_SIZE,

    # --- Lines & markers ---
    "lines.linewidth": 1.0,
    "lines.markersize": 0.7,

    # --- Ticks ---
    "xtick.labelsize": MY_FONT_SIZE,
    "ytick.labelsize": MY_FONT_SIZE,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.bottom": True,
    "ytick.left": True,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "axes.linewidth": 0.7,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,

    # Grid
    "axes.grid": False,
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.5,


    # --- Layout ---
    "figure.autolayout": True,

    # --- DPI ---
    "savefig.dpi": 600,
    "figure.dpi": 300,
})


#%%

simulation_condition = "3states"
project_root = Path(__file__).resolve().parents[2]
figure_dir = project_root / "figures" / simulation_condition
data_dir = project_root / "data" / "simulations"
data_file = data_dir / f"{simulation_condition}_data.npz"



os.makedirs(figure_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok = True)


#global settings (used across all the simulations)

# Set seeds for reproducibility.
seed_base = 2026

# Simulation parameters.
n_samples = 20

fs = 250         # Sampling frequency in Hz.
n_seconds = 60*3   # Total duration in seconds.
time_vec = np.arange(int(fs * n_seconds)) / fs


burst_amp_sigma = 0.1
tukey_alpha=0.25
state_transition = 'return_to_baseline'


beta = 1 #pink noise


# For burst segments, specify duration as the number of cycles.
burst_cycles = [3, 10]

# For noise segments, specify duration in seconds.
noise_duration = [0.2, 2]



# Condition 1: frequency distances (3 states)
freqs_dist_range = [[10, 40], [15, 40], [20, 40], [25, 40], [30, 40],[35, 40]]  #30, 25, 20, 15, 10, 5Hz diff

# Condition 2: Signal-to-noise ratio (in dB)
snr_range = np.arange(-10, 11, 2)  # varying SNRs from -10 to 10 dB


###################################################################

# Initialize variables
states_sample = np.zeros((n_samples, len(freqs_dist_range), len(snr_range), len(time_vec)))
bursts_sample = np.zeros((n_samples, len(freqs_dist_range), len(snr_range), len(time_vec)))
noise_sample = np.zeros((n_samples, len(freqs_dist_range), len(snr_range), len(time_vec)))
signal_sample = np.zeros((n_samples, len(freqs_dist_range), len(snr_range), len(time_vec)))


# Loop over the number of samples.
for sample_id in range(n_samples):

    # One seed per sample
    sample_seed = seed_base + sample_id


    # Loop over frequency conditions
    for freq_id, freqs in enumerate(freqs_dist_range):

        # Loop over SNR values.
        for snr_id, snr_db in enumerate(snr_range):

            # Reset RNG with the same sample seed for every condition
            # inside this sample.
            rng = np.random.default_rng(sample_seed)

            # Simulate a signal with sine bursts.

            signal_dict = simulate_bursty_signal(time_vec, fs, freqs, burst_cycles, noise_duration,
                                                 state_transition=state_transition, burst_type="sine", use_filter=True,
                                                 snr_db=snr_db, burst_amp_sigma=burst_amp_sigma, beta=beta,
                                                 tukey_alpha=tukey_alpha, rng=rng)

            states_sample[sample_id, freq_id, snr_id, :] = signal_dict["states"]
            bursts_sample[sample_id, freq_id, snr_id, :] =signal_dict["bursts"]
            noise_sample[sample_id, freq_id, snr_id, :] = signal_dict["noise"]
            signal_sample[sample_id, freq_id, snr_id, :] = signal_dict["signal"]



print("Signal shape", signal_sample.shape, "[samples, frequencies, SNRs, data points]")


### save the data

data_file = data_dir / f"{simulation_condition}_data.npz"
np.savez_compressed(data_file,
                    signal_sample=signal_sample,
                    states_sample=states_sample,
                    bursts_sample=bursts_sample,
                    noise_sample=noise_sample)

print(f"Data saved as {data_file}")



