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

from scipy.signal import welch
from matplotlib.colors import Normalize
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from nds_toolbox.sim.bursts.simulator import simulate_bursty_signal

# ============================================================
# Directories
# ============================================================
simulation_condition = "multiple_exponents"

figure_dir = f"../figures/{simulation_condition}"
data_dir = "../data/simulations"
data_file = f"{data_dir}/{simulation_condition}_data.npz"

os.makedirs(figure_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# ============================================================
# Simulation parameters
# ============================================================
seed_base = 2026
n_samples = 10

fs = 250
segment_seconds = 30
n_repetitions = 3

snr_db = 2
freq = 20

burst_amp_sigma = 0.1
tukey_alpha = 0.25
state_transition = "return_to_baseline"

burst_cycles = [3, 10]
noise_duration = [0.2, 2.0]

burst_type = "sine"

# condition: Aperiodic exponents
base_exponent = 1.0

variability_levels = np.arange(-1.0, 1.1, 0.2)
print(len(variability_levels))

exponent_range = [
    (base_exponent, np.round(base_exponent + delta, 1))
    for delta in variability_levels
    if not np.isclose(delta, 0.0)
]

print("Exponent pairs:", exponent_range)

# %%

# ============================================================
# Signal dimensions
# ============================================================
n_exponents = len(exponent_range[0])

segment_samples = int(round(fs * segment_seconds))

total_samples = (
        segment_samples
        * n_exponents
        * n_repetitions
)

time_vec = np.arange(segment_samples) / fs
total_time_vec = np.arange(total_samples) / fs

print("Segment duration:", segment_seconds, "s")
print("Total duration:", total_samples / fs, "s")
print("Total samples:", total_samples)


# ============================================================
# Simulation function
# ============================================================
def simulate_bursty_signal_with_multiple_exponents(
        exponents,
        segment_seconds,
        n_repetitions,
        seed=None,
):
    time_vec = np.arange(
        int(round(fs * segment_seconds))
    ) / fs

    n_exponents = len(exponents)

    signal_segments = []
    state_segments = []
    burst_segments = []
    noise_segments = []

    for repetition_id in range(n_repetitions):

        for exponent_id, exponent in enumerate(exponents):
            rng = np.random.default_rng(seed + repetition_id + exponent_id)

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
                noise_addition_mode="global_snr",
                burst_amp_sigma=burst_amp_sigma,
                beta=exponent,
                tukey_alpha=tukey_alpha,
                rng=rng,
            )

            signal = signal_dict["signal"]
            signal = signal - np.mean(signal) / np.std(signal)
            signal_segments.append(signal)
            burst_segments.append(signal_dict["bursts"])
            noise_segments.append(signal_dict["noise"])

            # Ground-truth state definitions:
            #
            # 0: Noise with exponent beta_0
            # 1: Noise with exponent beta_1
            # 2: Bursts + noise with exponent beta_0
            # 3: Bursts + noise with exponent beta_1

            states = np.where(
                signal_dict["states"] == 0,
                exponent_id,
                exponent_id + n_exponents,
            )

            state_segments.append(states)

    return {
        "signal": np.concatenate(signal_segments),
        "states": np.concatenate(state_segments),
        "bursts": np.concatenate(burst_segments),
        "noise": np.concatenate(noise_segments),
    }


# ============================================================
# Generate simulated datasets
# ============================================================
sample_shape = (
    n_samples,
    len(exponent_range),
    total_samples,
)

states_sample = np.empty(sample_shape, dtype=np.int32)
bursts_sample = np.empty(sample_shape)
noise_sample = np.empty(sample_shape)
signal_sample = np.empty(sample_shape)

for sample_id in tqdm(range(n_samples)):

    sample_seed = seed_base + sample_id

    for exponent_id, exponents in enumerate(exponent_range):
        # Common random numbers across exponent conditions
        # within each sample.

        signal_dict = (
            simulate_bursty_signal_with_multiple_exponents(
                exponents,
                segment_seconds,
                n_repetitions,
                seed=sample_seed,
            )
        )

        states_sample[sample_id, exponent_id, :] = (
            signal_dict["states"]
        )

        bursts_sample[sample_id, exponent_id, :] = (
            signal_dict["bursts"]
        )

        noise_sample[sample_id, exponent_id, :] = (
            signal_dict["noise"]
        )

        signal_sample[sample_id, exponent_id, :] = (
            signal_dict["signal"]
        )

print("Signal shape:", signal_sample.shape)
print("States shape:", states_sample.shape)

# ============================================================
# Save simulation data
# ============================================================
np.savez_compressed(
    data_file,
    signal_sample=signal_sample,
    states_sample=states_sample,
    bursts_sample=bursts_sample,
    noise_sample=noise_sample,
    exponent_range=np.asarray(exponent_range),
    variability_levels=variability_levels,
    fs=fs,
    segment_seconds=segment_seconds,
    n_repetitions=n_repetitions,
    seed_base=seed_base,
    snr_db=snr_db,
    freq=freq,
)

print(f"Data saved as {data_file}")


# ============================================================
# Compute exponent-specific PSDs
# ============================================================
def compute_mean_psd_by_exponent(
        data,
        condition_id,
        exponent_id,
):
    """Average segment-wise PSDs across samples and repetitions."""

    psds = []

    nperseg = min(2048, segment_samples)

    for sample_id in range(n_samples):

        for repetition_id in range(n_repetitions):
            start = (
                            repetition_id * n_exponents
                            + exponent_id
                    ) * segment_samples

            stop = start + segment_samples

            segment = data[
                      sample_id,
                      condition_id,
                      start:stop,
                      ]

            f, psd = welch(
                segment,
                fs=fs,
                nperseg=nperseg,
            )

            psds.append(psd)

    return f, np.mean(psds, axis=0)


# Reference PSD: beta = 1
f, reference_psd = compute_mean_psd_by_exponent(
    signal_sample,
    condition_id=0,
    exponent_id=0,
)

# PSD for the varying exponent in each condition
mean_psds = []

for condition_id in range(len(exponent_range)):
    f, psd = compute_mean_psd_by_exponent(
        signal_sample,
        condition_id=condition_id,
        exponent_id=1,
    )

    mean_psds.append(psd)

mean_psds = np.asarray(mean_psds)

# ============================================================
# Plot mean PSDs
# ============================================================
varying_exponents = np.array([
    exponents[1]
    for exponents in exponent_range
])

norm = Normalize(
    vmin=varying_exponents.min(),
    vmax=varying_exponents.max(),
)

cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(
    figsize=(FIG_WIDTH, 4),
)

fig.set_tight_layout(False)

# Reference spectrum
ax.plot(
    f,
    reference_psd,
    color="0.4",
    linestyle="--",
    linewidth=1.2,
    label=r"$\beta = 1.0$ (reference)",
)

# Varying exponents
for j, exponent in enumerate(varying_exponents):
    ax.plot(
        f,
        mean_psds[j],
        linewidth=1.0,
        color=cmap(norm(exponent)),
        label=rf"$\beta = {exponent:.1f}$",
    )

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlim(1, 80)

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD")
ax.set_title(f"{freq} Hz bursts")

fig.subplots_adjust(
    left=0.12,
    right=0.76,
    bottom=0.15,
    top=0.90,
)

ax.legend(
    title="Aperiodic exponent",
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    fontsize=MY_FONT_SIZE,
    title_fontsize=MY_FONT_SIZE,
)

file_path = os.path.join(
    figure_dir,
    "mean_psd_by_exponents.pdf",
)

fig.savefig(
    file_path,
    bbox_inches="tight",
    transparent=False,
)

plt.show()

# ============================================================
# Visualize example signal
# ============================================================
sample_id = 0

for condition_id in range(len(exponent_range)):
    fig, ax = plt.subplots(
        3, 1,
        figsize=(FIG_WIDTH, 5),
        sharex=True,
    )

    ax[0].plot(
        total_time_vec,
        signal_sample[sample_id, condition_id],
        color="black",
    )

    ax[0].set_ylabel("Signal (a.u.)")

    ax[1].plot(
        total_time_vec,
        bursts_sample[sample_id, condition_id],
        color="tab:blue",
    )

    ax[1].set_ylabel("Bursts")

    ax[2].step(
        total_time_vec,
        states_sample[sample_id, condition_id],
        where="post",
        color="black",
    )

    ax[2].set_ylabel("State")
    ax[2].set_xlabel("Time (s)")

    ax[2].set_yticks(np.arange(4))

    ax[2].set_yticklabels([
        rf"Noise $\beta={exponent_range[condition_id][0]:.1f}$",
        rf"Noise $\beta={exponent_range[condition_id][1]:.1f}$",
        rf"Burst $\beta={exponent_range[condition_id][0]:.1f}$",
        rf"Burst $\beta={exponent_range[condition_id][1]:.1f}$",
    ])

    # Show the transition between exponent segments
    ax[2].set_xlim(20, 40)

    plt.show()
