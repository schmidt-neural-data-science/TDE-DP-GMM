from nds_toolbox.sim.bursts.simulator import simulate_bursty_signal
from nds_toolbox.preprocessing.features import (compute_tde,trim_data)
from nds_toolbox.preprocessing.features import choose_embedding_dim
from nds_toolbox.utils.helper import compare_decoding_performance
import numpy as np
import pickle
import os


n_jobs = os.cpu_count()
print("n_jobs", n_jobs)

sim_cond = "4states"
data_dir = "../data/simulations"
data_file = f"{data_dir}/{sim_cond}_data.npz"


####
os.makedirs(data_dir, exist_ok = True)


# Set seeds for reproducibility.
seed = 2025
rng = np.random.default_rng(seed)

# ---parameters for burst simulator---
n_seconds = 60*3  # Total duration in seconds.

fs = 250
time_vec = np.linspace(0, n_seconds, int(fs * n_seconds))

snr_db = 2

freq = np.array([20, 30, 40])  #ground truth = 4 states (including noise state)


# For burst segments, specify duration as the number of cycles.
burst_cycles = [3, 7] #[3, 7]

# For noise segments, specify duration in seconds.
noise_duration = [0.5, 3] #[0.5, 3.]


burst_amp_sigma = 0.1
beta = 1 #pink noise


## simulate a bursty signal
signal_dict = simulate_bursty_signal(time_vec, fs, freq, burst_cycles, noise_duration, burst_type = "sine", use_filter= True, snr_db= snr_db, burst_amp_sigma = burst_amp_sigma, beta = beta, rng = rng)

signal, states, bursts, noise,  = signal_dict.values()


signal = (signal - np.mean(signal))/ np.std(signal) #standarize
num_emb = choose_embedding_dim(np.mean(freq), fs, min_cycles = 2.5, ensure_odd = True)
tde_signal = compute_tde(signal, num_emb)

trimmed_signal = trim_data(signal, num_emb, verbose = False)
trimmed_states = trim_data(states, num_emb, verbose = False)
trimmed_time_vec = np.linspace(0, len(trimmed_states)/fs, len(trimmed_states))


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



    return compare_decoding_performance(data_info = data_info, model_info = model_info,verbose = True)

performance_dir = "../data/performance"
os.makedirs(performance_dir, exist_ok=True)

from tqdm.auto import tqdm
total_run = len(num_states_vec)
bar = tqdm(num_states_vec, total=total_run)
all_results = [_run_one(num_states) for num_states in num_states_vec]



# Save to performance_dir
save_path = os.path.join(performance_dir, f"HMM_{sim_cond}_res.pkl")
with open(save_path, "wb") as f:
    pickle.dump(all_results, f)

print(f"Performance results saved as {save_path}")


