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

simulation_condition = "within_state_amplitude_variability"
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
# Condition: varying amplitude factor

base_amp_factor = 1.0

variability_levels = np.array([
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
])
# Each burst independently takes one of two amplitude factors.
amp_factor_pairs = [
    (
        base_amp_factor,
        base_amp_factor + delta,
    )
    for delta in variability_levels
]

# Maximum-to-minimum amplitude ratio
amp_ratios = np.array([
    max(factors) / min(factors)
    for factors in amp_factor_pairs
])

n_conditions = len(amp_factor_pairs)

print("Amplitude factor pairs:")
for factors, ratio in zip(amp_factor_pairs, amp_ratios):
    print(f"{factors}: ratio = {ratio:.2f}")

###################################################################


# Initialize variables
shape = (
    n_samples,
    n_conditions,
    len(time_vec),
)

states_sample = np.zeros(shape, dtype=np.int16)
bursts_sample = np.zeros(shape)
noise_sample = np.zeros(shape)
signal_sample = np.zeros(shape)

# Loop over samples
for sample_id in tqdm(
        range(n_samples),
        desc="Simulating",
):

    sample_seed = seed_base + sample_id

    # Generate reference signal with constant burst amplitude.
    # Scale noise to achieve the target SNR for this reference.
    reference_dict = simulate_bursty_signal(
        time_vec,
        fs,
        freq,
        burst_cycles,
        noise_duration,
        state_transition=state_transition,
        burst_type=burst_type,
        noise_addition_mode="global_snr",
        snr_db=snr_db,
        use_filter=True,
        burst_amp_sampling="choice",
        burst_amp_factors=(
            base_amp_factor,
            base_amp_factor,
        ),
        beta=beta,
        tukey_alpha=tukey_alpha,
        rng=np.random.default_rng(sample_seed),
    )

    # Fixed noise realization across conditions within this sample.
    reference_noise = reference_dict["noise"]
    reference_states = reference_dict["states"]

    for condition_id, amp_factors in enumerate(amp_factor_pairs):

        # Reset RNG to preserve the underlying burst
        # sequence across amplitude conditions.
        rng = np.random.default_rng(sample_seed)

        signal_dict = simulate_bursty_signal(
            time_vec,
            fs,
            freq,
            burst_cycles,
            noise_duration,
            state_transition=state_transition,
            burst_type=burst_type,
            noise_addition_mode="unscaled",
            custom_noise=reference_noise,
            burst_amp_sampling="choice",
            burst_amp_factors=amp_factors,
            beta=beta,
            tukey_alpha=tukey_alpha,
            rng=rng,
        )

        # Verify that the reference and experimental
        # signals have identical state sequences.
        if not np.array_equal(
                reference_states,
                signal_dict["states"],
        ):
            raise RuntimeError(
                "State sequences differ between the "
                "reference and experimental signals."
            )

        states_sample[sample_id, condition_id, :] = (
            signal_dict["states"]
        )

        bursts_sample[sample_id, condition_id, :] = (
            signal_dict["bursts"]
        )

        noise_sample[sample_id, condition_id, :] = (
            signal_dict["noise"]
        )

        signal_sample[sample_id, condition_id, :] = (
            signal_dict["signal"]
        )

print(
    "Signal shape:",
    signal_sample.shape,
    "[samples, amplitude conditions, time points]",
)

# Verify identical state sequences across conditions.
same_states = np.all(
    states_sample == states_sample[:, :1, :]
)

print(
    "Identical state sequences across conditions:",
    same_states,
)

if not same_states:
    raise RuntimeError(
        "Ground-truth state sequences differ across "
        "amplitude conditions."
    )

# Verify identical noise across conditions within each sample.
same_noise = np.allclose(
    noise_sample,
    noise_sample[:, :1, :],
)

print(
    "Identical noise across conditions:",
    same_noise,
)

if not same_noise:
    raise RuntimeError(
        "Noise differs across amplitude conditions."
    )

# Verify signal decomposition.
assert np.allclose(
    signal_sample,
    bursts_sample + noise_sample,
)

### save the data

data_file = f"{data_dir}/{simulation_condition}_data.npz"
np.savez_compressed(data_file,
                    signal_sample=signal_sample,
                    states_sample=states_sample,
                    bursts_sample=bursts_sample,
                    noise_sample=noise_sample)

print(f"Data saved as {data_file}")

# %%


# ============================================================
# Power spectral density
# ============================================================

# Welch supports multidimensional arrays.
# Input:  (samples, conditions, time)
# Output: (samples, conditions, frequencies)

f_psd, psds = welch(
    signal_sample,
    fs=fs,
    nperseg=fs * 2,
    axis=-1,
)

mean_psds = psds.mean(axis=0)

# ============================================================
# Plot mean PSD across amplitude variability conditions
# ============================================================

norm = Normalize(
    vmin=variability_levels.min(),
    vmax=variability_levels.max(),
)

cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(
    figsize=(FIG_WIDTH, 5),
)

fig.set_tight_layout(False)

for condition_id, delta in enumerate(variability_levels):
    ax.plot(
        f_psd,
        mean_psds[condition_id],
        linewidth=0.7,
        color=cmap(norm(delta)),
        label=f"{amp_ratios[condition_id]:.1f}",
        alpha=1,
    )

ax.set_yscale("log")
ax.set_title(f"{freq} Hz")

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD")

ax.set_xlim(1, 80)

fig.subplots_adjust(
    left=0.12,
    right=0.78,
    bottom=0.13,
    top=0.90,
)

ax.legend(
    title="Amplitude ratio",
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    fontsize=MY_FONT_SIZE,
    title_fontsize=MY_FONT_SIZE,
)

file_path = os.path.join(
    figure_dir,
    "mean_psd_by_burst_amp.pdf",
)

fig.savefig(
    file_path,
    bbox_inches="tight",
    transparent=False,
)

plt.show()

# ============================================================
# Plot example burst signals
# ============================================================

condition_indices = [0, 3, 5]

fig, axes = plt.subplots(
    len(condition_indices),
    1,
    figsize=(FIG_WIDTH, 4),
    sharex=True,
    sharey=True,
)

for ax, condition_id in zip(axes, condition_indices):
    ax.plot(
        time_vec,
        bursts_sample[0, condition_id],
        linewidth=0.8,
    )

    ax.set_ylabel("Amplitude")

    ax.set_title(
        f"Amplitude ratio = {amp_ratios[condition_id]:.1f}"
    )

axes[-1].set_xlabel("Time (s)")
axes[-1].set_xlim(0, 30)

plt.show()
