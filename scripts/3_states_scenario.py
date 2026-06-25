
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


#%%
sim_cond = "3states"
signal_dir = "../data/simulations"
performance_dir = "../data/performance"

sim_data = np.load(f'{signal_dir}/{sim_cond}_data.npz', allow_pickle=True)
signal_sample = sim_data['signal_sample']
states_sample = sim_data['states_sample']
print("shape of signal_sample", signal_sample.shape)

seed_base = 2026
fs = 250
num_states = 3

# Condition 1: frequency distances (3 states)
freqs_dist_range = [[10, 40], [15, 40], [20, 40], [25, 40], [30, 40], [35, 40]]  # 30, 25, 20, 15, 10, 5Hz diff

snr_range = np.arange(-10, 11, 2)


#%%

def _run_one(sample_id, freq_id, snr_id):
    freq = freqs_dist_range[freq_id]
    freq_for_emb = float(np.mean(freq))

    # Different embeddings for each model
    num_emb_dpgmm = choose_embedding_dim(
        freq_for_emb,
        fs,
        min_cycles=2.5,
        ensure_odd=True,
    )

    num_emb_hmm = choose_embedding_dim(
        freq_for_emb,
        fs,
        min_cycles=3.0,
        ensure_odd=True,
    )


    snr = snr_range[snr_id]
    sig = signal_sample[sample_id, freq_id, snr_id]
    sig = (sig - np.mean(sig)) / np.std(sig)  # standardize

    st = states_sample[sample_id, freq_id, snr_id].astype(int)

    # use different seed for each sample
    seed = int(seed_base + sample_id)

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

        "num_emb": num_emb_dpgmm, #this is just fall back
        # Method-specific embeddings
        "num_emb_dpgmm": num_emb_dpgmm,
        "num_emb_hmm": num_emb_hmm,

        "use_model_tqdm": False,
        "use_thresholding": True,
        "filter_freq": np.array(freq),
        "imputing_spurious_states": True,
        "compute_summary_stats": False,
        "min_samples": int(np.round((2 / np.max(freq)) * fs)),
        "truncate_weights": False,
        "debug_mode": False,
        "n_jobs": inner_n_jobs
    }

    return compare_decoding_performance(data_info=data_info, model_info=model_info, verbose=True)



# Build the parameter grid
param_iter = itertools.product(
    range(signal_sample.shape[0]),
    range(signal_sample.shape[1]),
    range(signal_sample.shape[2])
)


print("num sims: ", int(signal_sample.size / signal_sample.shape[-1]))


from tqdm.auto import tqdm
total_sims = signal_sample.shape[0] * signal_sample.shape[1] * signal_sample.shape[2]
bar = tqdm(param_iter, total=total_sims, desc="Running sims", unit="sim")

all_results = [_run_one(i, j, k) for i, j, k in bar]

# Save
os.makedirs(performance_dir, exist_ok=True)
save_path = os.path.join(performance_dir, f"results_{sim_cond}.pkl")
with open(save_path, "wb") as f:
    pickle.dump(all_results, f)
print(f"Performance results saved as {save_path}")

