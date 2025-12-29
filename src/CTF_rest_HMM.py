
from nds_toolbox.preprocessing.features import choose_embedding_dim
from nds_toolbox.utils.helper import compare_decoding_performance
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


from scipy.signal import welch
psd_sample = []
for data in data_all:
    f, psd = welch(data.T, fs, scaling="density", average="mean",nperseg=fs * 2)
    psd_sample.append(psd)

psd_sample = np.array(psd_sample)  # (65, 1, 51)
mean_psd = np.mean(psd_sample, axis=0)[0]

from scipy.signal import find_peaks
distance_bins = int(4/(f[1] - f[0]))
peaks, _ = find_peaks(mean_psd, distance= distance_bins)

print(peaks)
peaks = peaks[1:] #manually remove 2 hz
print(f[peaks])




#%% preprocessing
signal = np.concatenate(data_all)
signal = (signal - np.mean(signal)) / np.std(signal) #standarize the signal
signal = signal[:, 0]

num_emb = choose_embedding_dim(np.mean(f[peaks]), fs, min_cycles = 2.5, ensure_odd = True)

print("num_embeddings", num_emb) #15 emb




num_states_vec = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]



#%%

def _run_one(num_states):
    data_info = {
        "signal": signal,
        "true_states": None,
        "burst_f": None,
        "fs": fs,
        "snr": None,}



    model_info = {
        "seed": seed,
        "use_dpgmm": False,
        "use_hmm": True,
        "num_states": num_states,
        "num_emb": num_emb,
        "use_model_tqdm": True,
        "use_thresholding": False,
        "filter_freq": None,
        "imputing_spurious_states":True,
        "compute_summary_stats": True,
        "min_samples": np.round((2/f[peaks.max()])*fs).astype(int),
        "truncate_weights": False,
        "debug_mode": False,
        "n_jobs": n_jobs
    }


    return compare_decoding_performance(data_info = data_info, model_info = model_info,verbose = True)




performance_dir = "../data/CTF_rest"
os.makedirs(performance_dir, exist_ok=True)


from tqdm.auto import tqdm
total_run = len(num_states_vec)
bar = tqdm(num_states_vec, total=total_run)

all_results = [_run_one(num_states) for num_states in num_states_vec]



# Save to performance_dir
save_path = os.path.join(performance_dir, f"CTF_rest_HMM_result.pkl")
with open(save_path, "wb") as f:
    pickle.dump(all_results, f)

print(f"Performance results saved as {save_path}")

