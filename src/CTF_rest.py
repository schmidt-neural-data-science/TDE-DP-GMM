
import matplotlib.pyplot as plt

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
    "axes.linewidth": 0.7,
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

from nds_toolbox.sim.bursts.simulator import simulate_bursty_signal
from nds_toolbox.preprocessing.features import (compute_tde,trim_data)
from nds_toolbox.preprocessing.features import choose_embedding_dim
from nds_toolbox.utils.helper import compare_decoding_performance
import numpy as np
import pickle
import os
import numpy as np



signal_dir = "../data/CTF_rest"
files = [f"{signal_dir}/bursts/subject{i:02d}.npy" for i in range(65)]
data_all = [np.load(f, allow_pickle=True) for f in files]


fs = 100
seed = 2025

n_jobs = os.cpu_count()
print("n_jobs:", n_jobs)

#%%

from scipy.signal import welch
import matplotlib.pyplot as plt
psd_sample = []
for data in data_all:
    f, psd = welch(data.T, fs)
    psd_sample.append(psd)

psd_sample = np.array(psd_sample)  # (65, 1, 51)
mean_psd = np.mean(psd_sample, axis=0)[0]


print(peaks)
peaks = peaks[1:] #manually remove 2 hz
print(f[peaks])


plt.plot(f, mean_psd, "k")
plt.plot(f[peaks], mean_psd[peaks], "rx")

plt.xlabel("Freqeuncy (Hz)")
plt.ylabel("PSD (V**2/Hz)")
plt.show()


#%% preprocessing
signal = np.concatenate(data_all)
signal = (signal - np.mean(signal)) / np.std(signal) #standarize the signal
signal = signal[:, 0]

num_emb = choose_embedding_dim(np.mean(f[peaks]), fs, min_cycles = 2.5, ensure_odd = True)

print("num_embeddings", num_emb) #15 emb

"""
[ 5.5 11.  17.  21.  31.5]
Chosen window size = 0.145 (sec): minimum cycles of 2.5 at 17.2 Hz
Chosen embedding dimension = 15: sampling rate = 100 Hz
num_embeddings 15
num_states:  15
"""

#alpha will be bayesian estimated, but to roughly estimate num_states beforehand, i set it to 1 (uniform stickbreaking beta distribution)
num_states = int(np.ceil(np.log(len(signal)))) # E[K] = a ln n
print("num_states: ", num_states)
# this sets K above the expected occupied clusters
#15 states



#%%
data_info = {
    "signal": signal,
    "true_states": None,
    "burst_f": None,
    "fs": fs,
    "snr": None,}





model_info = {
    "seed": seed,
    "use_dpgmm": True,
    "use_hmm": False,
    "num_states": num_states,
    "num_emb": num_emb,
    "use_model_tqdm": True,
    "use_thresholding": False,
    "filter_freq": None,
    "imputing_spurious_states":True,
    "compute_summary_stats": True,
    "min_samples":np.round((2/f[peaks.max()])*fs).astype(int),
    "truncate_weights": True,
    "debug_mode": False,
    "n_jobs": n_jobs
}


res = compare_decoding_performance(data_info = data_info, model_info = model_info,verbose = True)


performance_dir = "../data/CTF_rest"
os.makedirs(performance_dir, exist_ok=True)


# Save to performance_dir
save_path = os.path.join(performance_dir, "CTF_rest_result.pkl")
with open(save_path, "wb") as f:
    pickle.dump(res, f)

print(f"Performance results saved as {save_path}")

