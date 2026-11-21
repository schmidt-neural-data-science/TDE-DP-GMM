from nds_toolbox.sim.bursts.simulator import simulate_bursty_signal
from nds_toolbox.preprocessing.features import (compute_tde,trim_data)
from nds_toolbox.preprocessing.features import choose_embedding_dim
from nds_toolbox.utils.helper import compare_decoding_performance
import numpy as np
import pickle
import os

import os
import pickle
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator

from nds_toolbox.preprocessing.features import (
    choose_embedding_dim,
    compute_wavelet,
    trim_data,
)
from nds_toolbox.sim.bursts.simulator import simulate_bursty_signal
from nds_toolbox.utils.helper import compare_decoding_performance

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
    "figure.autolayout": False,  # Use constrained_layout below.

    # --- DPI ---
    "savefig.dpi": 900,
    "figure.dpi": 300,
})





# %%

# Find the project root in both script and interactive execution.
start_dir = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd().resolve()
)
for candidate in (start_dir, *start_dir.parents):
    if (candidate / "nds_toolbox").is_dir() and (candidate / "data").is_dir():
        project_root = candidate
        break
else:
    raise FileNotFoundError(f"Could not find the TDE-DP-GMM project root from {start_dir}")


#%%
n_jobs = 5
print("n_jobs", n_jobs)

sim_cond = "3states_overparameterized_hmm"


data_dir = project_root / "data" / "simulations"
data_file = data_dir / f"{sim_cond}_data.npz"

fig_dir = project_root / "figures" / sim_cond
fig_dir.mkdir(parents=True, exist_ok=True)


####
data_dir.mkdir(parents=True, exist_ok=True)


# Set seeds for reproducibility.
seed = 2026
rng = np.random.default_rng(seed)

# ---parameters for burst simulator---
n_seconds = 60 * 3  # Total duration in seconds.
fs = 250
time_vec = np.arange(int(fs * n_seconds)) / fs

snr_db = 2

freq = np.array([20, 40])  # Three true states, including the noise state.


burst_amp_sigma = 0.1
tukey_alpha = 0.25
state_transition = "return_to_baseline"


beta = 1  # Pink noise.


# For burst segments, specify duration as the number of cycles.
burst_cycles = [3, 10]

# For noise segments, specify duration in seconds.
noise_duration = [0.2, 2.]


## simulate a bursty signal
signal_dict = simulate_bursty_signal(time_vec, fs, freq, burst_cycles, noise_duration, state_transition = state_transition, burst_type = "sine", use_filter= True, snr_db= snr_db, burst_amp_sigma = burst_amp_sigma, beta = beta, tukey_alpha=tukey_alpha, rng = rng)

signal, states, bursts, noise,  = signal_dict.values()


signal = (signal - np.mean(signal))/ np.std(signal) #standarize

#embeddings for HMM
num_emb = choose_embedding_dim(np.mean(freq), fs, min_cycles = 2.5, ensure_odd = True)
tde_signal = compute_tde(signal, num_emb)

trimmed_signal = trim_data(signal, num_emb, verbose = False)
trimmed_states = trim_data(states, num_emb, verbose = False)
trimmed_time_vec = np.arange(len(trimmed_states)) / fs


### save the data

np.savez_compressed(data_file,
                    signal_sample=signal,
                    states_sample=states,
                    bursts_sample=bursts,
                    noise_sample=noise)

print(f"Data saved as {data_file}")



###






num_states_vec = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]



def _run_one(num_states):
    data_info = {
        "signal": signal,
        "true_states": None,
        "burst_f": freq,
        "fs": fs,
        "snr": snr_db,}


    model_info = {
        "seed": seed,
        "use_dpgmm": False,
        "use_hmm": True,
        "num_states": num_states,
        "num_emb": num_emb,
        "use_model_tqdm": True,
        "use_thresholding": False,
        "filter_freq": None,
        "imputing_spurious_states":False,
        "compute_summary_stats": False,
        "min_samples": None,
        "truncate_weights": False,
        "debug_mode": False,
        "n_jobs": n_jobs
    }



    return compare_decoding_performance(
        data_info=data_info,
        model_info=model_info,
        metrics=("mcc", "ari", "kappa"),
        verbose=True,
    )

performance_dir = "../../data/performance"
os.makedirs(performance_dir, exist_ok=True)

from tqdm.auto import tqdm
total_run = len(num_states_vec)
bar = tqdm(num_states_vec, total=total_run)
all_results = [_run_one(num_states) for num_states in num_states_vec]



# Save to performance_dir
save_path = os.path.join(performance_dir, f"{sim_cond}_est_k_res.pkl")
with open(save_path, "wb") as f:
    pickle.dump(all_results, f)

print(f"Performance results saved as {save_path}")




