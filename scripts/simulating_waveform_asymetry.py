from scipy.signal import welch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from nds_toolbox.sim.bursts.simulator import simulate_bursty_signal
from matplotlib.colors import Normalize

# %%


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

# %%

simulation_condition = "waveform_asymmetry"
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
time_vec = np.linspace(0, n_seconds, int(fs * n_seconds))

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
burst_type = "asine"
# Condition 1: varying waveform asymmetry
rise_ratio_range = np.arange(0.1, 1, 0.1)  # varying waveform asymmetry from 0.1 to 0.9
# rise_ratio_sampling = "choice"  # "choice" or "uniform"

print(rise_ratio_range)
###################################################################

# Initialize variables
states_sample = np.zeros((n_samples, len(rise_ratio_range), len(time_vec)))
bursts_sample = np.zeros((n_samples, len(rise_ratio_range), len(time_vec)))
noise_sample = np.zeros((n_samples, len(rise_ratio_range), len(time_vec)))
signal_sample = np.zeros((n_samples, len(rise_ratio_range), len(time_vec)))

# Loop over the number of samples.
for sample_id in range(n_samples):

    # One seed per sample
    sample_seed = seed_base + sample_id

    # Loop over width values.
    for width_id, rise_ratio in enumerate(rise_ratio_range):
        # Reset RNG with the same sample seed for every condition
        # inside this sample.
        rng = np.random.default_rng(sample_seed)

        # Simulate a signal with sine bursts.

        signal_dict = simulate_bursty_signal(time_vec, fs, freq, burst_cycles, noise_duration,
                                             state_transition=state_transition, burst_type=burst_type,
                                             rise_ratio=rise_ratio, use_filter=True,
                                             snr_db=snr_db, burst_amp_sigma=burst_amp_sigma, beta=beta,
                                             tukey_alpha=tukey_alpha, rng=rng)

        states_sample[sample_id, width_id, :] = signal_dict["states"]
        bursts_sample[sample_id, width_id, :] = signal_dict["bursts"]
        noise_sample[sample_id, width_id, :] = signal_dict["noise"]
        signal_sample[sample_id, width_id, :] = signal_dict["signal"]

# %%
print("Signal shape", signal_sample.shape, "[samples, widths, data points]")

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
    for snr_id in range(signal_sample.shape[1]):
        f, psd = welch(signal_sample[sample_id, snr_id, :], fs)
        psds.append(np.array(psd))

psds = np.array(psds).reshape(signal_sample.shape[0], signal_sample.shape[1], -1)

mean_psds = psds.mean(axis=0)

norm = Normalize(
    vmin=min(rise_ratio_range),
    vmax=max(rise_ratio_range),
)
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(
    1, 1,
    figsize=(FIG_WIDTH, 5),
)

# Disable automatic layout to allow manual adjustments
fig.set_tight_layout(False)

for j, rise_ratio in enumerate(rise_ratio_range):
    ax.plot(
        f,
        mean_psds[j, :],
        linewidth=0.7,
        color=cmap(norm(rise_ratio)),
        label=f"{rise_ratio:.1f}",
        alpha=1,
    )

ax.set_yscale("log")
ax.set_title(f"{freq} Hz")
ax.set_xlabel("Frequency (Hz)")
ax.set_xlim(1, 80)
ax.set_ylabel("PSD")

# Leave sufficient space for the legend
fig.subplots_adjust(
    left=0.12,
    right=0.78,
    bottom=0.13,
    top=0.90,
)

# Position legend inside the figure canvas
handles, labels = ax.get_legend_handles_labels()

ax.legend(
    handles,
    labels,
    title="Varying rise ratios",
    frameon=True,
    loc="upper right",
    ncol=2,
    bbox_to_anchor=(0.98, 0.98),
    fontsize=MY_FONT_SIZE,
    title_fontsize=MY_FONT_SIZE,
)

# Save figure
file_path = os.path.join(
    figure_dir,
    "mean_psd_by_waveform.pdf",
)

fig.savefig(
    file_path,
    bbox_inches="tight",
    transparent=False,
)

plt.show()

# %%

fig, ax = plt.subplots(3, 1, figsize=(FIG_WIDTH, 5), sharex=True)
ax[0].plot(time_vec, signal_sample[0, 0])
ax[0].plot(time_vec, 1.2 + bursts_sample[0, 0])
ax[0].set_xlim(0, 5)
plt.show()
