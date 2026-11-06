import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import welch, find_peaks

from nds_toolbox.preprocessing.features import choose_embedding_dim
from nds_toolbox.utils.helper import compare_decoding_performance

# %%

FIG_WIDTH = 7.5
MY_FONT_SIZE = 9

plt.rcParams.update({
    # Fonts
    "font.family": "Arial",
    "font.size": MY_FONT_SIZE,

    # Legend
    "legend.fontsize": MY_FONT_SIZE,
    "legend.frameon": False,

    # Titles
    "figure.titlesize": MY_FONT_SIZE,
    "axes.titleweight": "normal",
    "axes.titlesize": MY_FONT_SIZE,

    # Axes
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.7,
    "axes.unicode_minus": True,
    "axes.labelsize": MY_FONT_SIZE,

    # Lines and markers
    "lines.linewidth": 1.0,
    "lines.markersize": 0.7,

    # Ticks
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
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,

    # Grid
    "axes.grid": False,
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.5,

    # Layout
    "figure.autolayout": True,

    # DPI
    "savefig.dpi": 600,
    "figure.dpi": 300,
})

# %% Load data

signal_dir = "../data/CTF_rest"

files = [
    f"{signal_dir}/bursts/subject{i:02d}.npy"
    for i in range(65)
]

data_all = [
    np.load(f, allow_pickle=True)
    for f in files
]

fs = 100
seed = 2026

n_jobs_all = os.cpu_count() or 1
n_jobs = 2

print("Available cores:", n_jobs_all)
print("n_jobs:", n_jobs)

# %% Identify spectral peaks

from scipy.signal import welch
import matplotlib.pyplot as plt

psd_sample = []
for data in data_all:
    f, psd = welch(data.T, fs)
    psd_sample.append(psd)

psd_sample = np.array(psd_sample)  # (65, 1, 51)
mean_psd = np.mean(psd_sample, axis=0)[0]

from scipy.signal import find_peaks

distance_bins = int(4 / (f[1] - f[0]))
peaks, _ = find_peaks(mean_psd, distance=distance_bins)

print(peaks)
peaks = peaks[1:]  # manually remove 2 hz
print(f[peaks])

plt.plot(f, mean_psd, "k")
plt.plot(f[peaks], mean_psd[peaks], "rx")

plt.xlabel("Freqeuncy (Hz)")
plt.ylabel("PSD (V**2/Hz)")
plt.show()

# %% Preprocessing

# Concatenate subjects and select channel 0.
signal = np.concatenate(data_all, axis=0)[:, 0]

# Standardize signal.
signal = (
                 signal - np.mean(signal)
         ) / np.std(signal)

print("Signal shape:", signal.shape)

# %% Model parameters

# Approximate expected number of occupied clusters
# for a Dirichlet process with concentration alpha = 1.
# E[K] ≈ alpha * log(n)
num_states = int(np.ceil(np.log(len(signal))))

print("num_states:", num_states)

# Keep min_samples fixed across embedding dimensions.
min_samples = np.round((2 / f[peaks.max()]) * fs).astype(int)

print("min_samples:", min_samples)

# %% Evaluate embedding dimensions

all_results = []
embedding_dims = [11, 13, 15, 17, 19, 21]

for num_emb in embedding_dims:
    print(
        f"Number of embeddings: {num_emb}"
    )

    data_info = {
        "signal": signal,
        "true_states": None,
        "burst_f": None,
        "fs": fs,
        "snr": None,
    }

    model_info = {
        "seed": seed,
        "use_dpgmm": True,
        "use_hmm": False,
        "num_states": num_states,
        "num_emb": num_emb,
        "use_model_tqdm": True,
        "use_thresholding": False,
        "filter_freq": None,
        "imputing_spurious_states": True,
        "remove_edge_states": True,
        "compute_summary_stats": True,
        "min_samples": min_samples,
        "truncate_weights": True,
        "debug_mode": False,
        "n_jobs": n_jobs,
    }

    res = compare_decoding_performance(
        data_info=data_info,
        model_info=model_info,
        metrics=("mcc", "ari", "kappa"),
        verbose=True,
    )

    all_results.append(res)

# %% Save results

performance_dir = "../data/CTF_rest"
os.makedirs(performance_dir, exist_ok=True)

save_path = os.path.join(
    performance_dir,
    "CTF_rest_result_emb_sensitivity.pkl",
)

with open(save_path, "wb") as file:
    pickle.dump(all_results, file)

print(f"Performance results saved as {save_path}")
