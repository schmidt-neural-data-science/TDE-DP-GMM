from scipy.signal import welch
import numpy as np
import os
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
figure_dir  = f"../figures/{simulation_condition}"
data_dir = "../data/simulations"
data_file = f"../data/simulations/{simulation_condition}_data.npz"


os.makedirs(figure_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok = True)


#global settings (used across all the simulations)

# Set seeds for reproducibility.
seed = 2025
rng = np.random.default_rng(seed)

# Simulation parameters.
n_samples = 5

fs = 250         # Sampling frequency in Hz.
n_seconds = 60*3   # Total duration in seconds.
time_vec = np.linspace(0, n_seconds, int(fs * n_seconds))

burst_amp_sigma = 0.1
beta = 1 #pink noise


# For burst segments, specify duration as the number of cycles.
burst_cycles = [3, 7]

# For noise segments, specify duration in seconds.
noise_duration = [0.5, 3.]




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


    # Loop over frequency conditions
    for freq_id, freqs in enumerate(freqs_dist_range):

        # Loop over SNR values.
        for snr_id, snr in enumerate(snr_range):
            # Simulate a signal with sine bursts.
            signal_dict = simulate_bursty_signal(
                time_vec, fs, freqs, burst_cycles, noise_duration,
                burst_type= "sine",
                snr_db=snr,
                beta= beta,
                burst_amp_sigma = burst_amp_sigma,
                rng = rng
            )
            states_sample[sample_id, freq_id, snr_id, :] = signal_dict["states"]
            bursts_sample[sample_id, freq_id, snr_id, :] =signal_dict["bursts"]
            noise_sample[sample_id, freq_id, snr_id, :] = signal_dict["noise"]
            signal_sample[sample_id, freq_id, snr_id, :] = signal_dict["signal"]



print("Signal shape", signal_sample.shape, "[samples, frequencies, SNRs, data points]")


### save the data

data_file = f"{data_dir}/{simulation_condition}_data.npz"
np.savez_compressed(data_file,
                    signal_sample=signal_sample,
                    states_sample=states_sample,
                    bursts_sample=bursts_sample,
                    noise_sample=noise_sample)

print(f"Data saved as {data_file}")


psds = []

for sample_id in range(signal_sample.shape[0]):
    for freq_id in range(signal_sample.shape[1]):
        for snr_id in range(signal_sample.shape[2]):
            f, psd = welch(signal_sample[sample_id, freq_id, snr_id, :], fs)
            psds.append(np.array(psd))

psds = np.array(psds).reshape(signal_sample.shape[0], signal_sample.shape[1], signal_sample.shape[2], -1)



mean_psds = psds.mean(axis=0)


norm  = Normalize(vmin=min(snr_range), vmax=max(snr_range))
cmap  = plt.get_cmap("viridis")

n_cols = len(freqs_dist_range)
fig, axes = plt.subplots(1, n_cols, figsize=(FIG_WIDTH, 2), sharex=True, sharey=True)

for col_idx, freq in enumerate(freqs_dist_range):
    ax = axes[col_idx]
    for j, snr in enumerate(snr_range):
        ax.plot(
            f,
            mean_psds[col_idx, j, :],
            linewidth=0.7,
            color=cmap(norm(snr)),
            label=f"{snr} dB",
            alpha = 1
        )
    ax.set_yscale("log")
    ax.set_title(f"{freq[0]} Hz & {freq[1]} Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_xlim(1, 80)
    if col_idx == 0:
        ax.set_ylabel(f"Log PSD")

# 3) single legend to the right
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    title="SNR (dB)",
    loc="center left",
    bbox_to_anchor=(1.01, 0.5),
    frameon=False,
    fontsize="large"
)

# 4) layout adjustment
plt.tight_layout()
plt.subplots_adjust(right=1)

#### save the figures

file_path = os.path.join(figure_dir, "mean_psd_by_snr.pdf")
fig.savefig(file_path,
            bbox_inches = "tight",
            transparent = False)



plt.show()
