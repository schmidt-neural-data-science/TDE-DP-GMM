import os
import numpy as np
import pickle
from nds_toolbox.preprocessing.features import choose_embedding_dim
from nds_toolbox.utils.helper import compare_decoding_performance


from joblib import Parallel, delayed
import itertools

#%%

total_cores = os.cpu_count() or 1
inner_n_jobs = total_cores


print("total_cores:", total_cores)
print("inner_n_jobs:", inner_n_jobs)


sim_cond = "2states"
signal_dir = "../data/simulations"
performance_dir = "../data/performance"


sim_data = np.load(f'{signal_dir}/{sim_cond}_data.npz', allow_pickle=True)
signal_sample = sim_data['signal_sample']
states_sample = sim_data['states_sample']
print("shape of signal_sample", signal_sample.shape)


#%%

signal_dir = "../data/simulations"
performance_dir = "../data/performance"




sim_data = np.load(f'{signal_dir}/{sim_cond}_data.npz', allow_pickle=True)
signal_sample = sim_data['signal_sample']
states_sample = sim_data['states_sample']

print("shape of signal_sample", signal_sample.shape)
n_samples = signal_sample.shape[0]
n_freqs   = signal_sample.shape[1]
n_fss     = signal_sample.shape[2]


#%%


##%%
seed_base = 2025
snr = 2
num_states = 2

freq_range = [10, 20, 30, 40, 50]
fs_range = [100, 250, 500]
num_embeddings_range = np.arange(3, 52, 4)
print(num_embeddings_range)

#%%


def _run_one(sample_id, freq_id, fs_id, emb_id):
    freq = freq_range[freq_id]
    fs = fs_range[fs_id]

    num_emb = num_embeddings_range[emb_id]

    sig = signal_sample[sample_id, freq_id, fs_id]
    sig = (sig - np.mean(sig)) / np.std(sig)  # standardize

    st = states_sample[sample_id, freq_id, fs_id].astype(int)

    # use different seed for each condition
    seed = int(seed_base + (sample_id * 10000 + freq_id * 100 + fs_id*10 + emb_id) * 997)

    data_info = {
        "signal": sig,
        "true_states": st,
        "burst_f": freq,
        "fs": fs,
        "snr": snr,
    }

    model_info = {
        "seed": seed,
        "use_dpgmm": True,
        "use_hmm": True,
        "num_states": num_states,
        "num_emb": num_emb,
        "use_model_tqdm": False,
        "use_thresholding": False,
        "filter_freq": None,
        "imputing_spurious_states": False,
        "compute_summary_stats": False,
        "min_samples": None,
        "truncate_weights": False,
        "debug_mode": False,
        "n_jobs": inner_n_jobs
    }

    return compare_decoding_performance(data_info=data_info, model_info=model_info, verbose=True)

# Build the parameter grid
param_iter = itertools.product(
    range(signal_sample.shape[0]),
    range(signal_sample.shape[1]),
    range(signal_sample.shape[2]),
    range(len(num_embeddings_range))
)






from tqdm.auto import tqdm

total_sims = signal_sample.shape[0] * signal_sample.shape[1] * signal_sample.shape[2] * len(num_embeddings_range)
print("num sims: ", total_sims)
bar = tqdm(param_iter, total=total_sims, desc="Running sims", unit="sim")

all_results = [_run_one(i, j, k, l) for i, j, k, l in bar]

# Save
os.makedirs(performance_dir, exist_ok=True)
save_path = os.path.join(performance_dir, f"results_{sim_cond}.pkl")
with open(save_path, "wb") as f:
    pickle.dump(all_results, f)
print(f"Performance results saved as {save_path}")