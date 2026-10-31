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

simulation_condition = "waveform_asymmetry_snr"
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

rise_ratio = [0.1, 0.9]  # waveform asymmetry will be sampled Uniform distribution between 0.1 to 0.9
rise_ratio_sampling = "uniform"

# Condition 1: burst morphologies
burst_types = ["sine", "asine"]

# Condition 2: Signal-to-noise ratio (in dB)
snr_range = np.arange(-10, 11, 2)  # varying SNRs from -10 to 10 dB

###################################################################

# Initialize variables
states_sample = np.zeros((n_samples, len(burst_types), len(snr_range), len(time_vec)))
bursts_sample = np.zeros((n_samples, len(burst_types), len(snr_range), len(time_vec)))
noise_sample = np.zeros((n_samples, len(burst_types), len(snr_range), len(time_vec)))
signal_sample = np.zeros((n_samples, len(burst_types), len(snr_range), len(time_vec)))

# Loop over the number of samples.
for sample_id in range(n_samples):

    # One seed per sample
    sample_seed = seed_base + sample_id

    for burst_type_id, burst_type in enumerate(burst_types):

        for snr_id, snr_db in enumerate(snr_range):
            # Reset RNG with the same sample seed for every condition
            # inside this sample.
            rng = np.random.default_rng(sample_seed)

            # Simulate a signal with sine bursts.

            signal_dict = simulate_bursty_signal(time_vec, fs, freq, burst_cycles, noise_duration,
                                                 state_transition=state_transition, burst_type=burst_type,
                                                 rise_ratio_sampling=rise_ratio_sampling, rise_ratio=rise_ratio,
                                                 use_filter=True,
                                                 snr_db=snr_db, burst_amp_sigma=burst_amp_sigma, beta=beta,
                                                 tukey_alpha=tukey_alpha, rng=rng)

            states_sample[sample_id, burst_type_id, snr_id, :] = signal_dict["states"]
            bursts_sample[sample_id, burst_type_id, snr_id, :] = signal_dict["bursts"]
            noise_sample[sample_id, burst_type_id, snr_id, :] = signal_dict["noise"]
            signal_sample[sample_id, burst_type_id, snr_id, :] = signal_dict["signal"]

# %%
print("Signal shape", signal_sample.shape, "[samples, burst_types, rise ratios, data points]")

### save the data

data_file = f"{data_dir}/{simulation_condition}_data.npz"
np.savez_compressed(data_file,
                    signal_sample=signal_sample,
                    states_sample=states_sample,
                    bursts_sample=bursts_sample,
                    noise_sample=noise_sample)

print(f"Data saved as {data_file}")

# %%


from matplotlib.gridspec import GridSpec

# ============================================================
# Compute PSD
# ============================================================

freqs_psd, psds = welch(
    signal_sample,
    fs=fs,
    nperseg=1024,
    axis=-1,
)

# Average across simulation samples
# Shape: (burst_types, SNRs, frequencies)
mean_psds = psds.mean(axis=0)

# ============================================================
# Plot mean PSD: 1 × 3 panels + shared colorbar
# ============================================================

norm = Normalize(
    vmin=snr_range.min(),
    vmax=snr_range.max(),
)
cmap = plt.get_cmap("viridis")

# Disable automatic layout for GridSpec
with plt.rc_context({"figure.autolayout": False}):
    fig = plt.figure(figsize=(FIG_WIDTH, 2.5))

    gs = GridSpec(
        1, 4,
        figure=fig,
        width_ratios=[1, 1, 1, 0.06],
        left=0.09,
        right=0.95,
        bottom=0.20,
        top=0.88,
        wspace=0.25,
    )

    # Initialize axes explicitly
    ax0 = fig.add_subplot(gs[0, 0])

    axes = [
        ax0,
        fig.add_subplot(gs[0, 1], sharex=ax0, sharey=ax0),
        fig.add_subplot(gs[0, 2], sharex=ax0, sharey=ax0),
    ]

    for burst_type_id, burst_type in enumerate(burst_types):

        ax = axes[burst_type_id]

        for snr_id, snr_db in enumerate(snr_range):
            ax.plot(
                freqs_psd,
                mean_psds[burst_type_id, snr_id, :],
                linewidth=0.8,
                color=cmap(norm(snr_db)),
            )

        ax.set_yscale("log")
        ax.set_title(burst_type.capitalize())
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xlim(1, 80)

        if burst_type_id > 0:
            ax.tick_params(axis="y", labelleft=False)

    axes[0].set_ylabel("PSD")

    # ============================================================
    # Shared colorbar
    # ============================================================

    cax = fig.add_subplot(gs[0, 3])

    sm = plt.cm.ScalarMappable(
        norm=norm,
        cmap=cmap,
    )

    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("SNR (dB)")
    cbar.set_ticks(snr_range[::2])

    # ============================================================
    # Save
    # ============================================================

    file_path = os.path.join(
        figure_dir,
        "mean_psd_by_waveform.svg",
    )

    fig.savefig(
        file_path,
        bbox_inches="tight",
        transparent=False,
    )

    plt.show()
