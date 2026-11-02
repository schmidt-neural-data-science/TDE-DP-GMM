import itertools
import os
import pickle

import numpy as np

from nds_toolbox.preprocessing.features import choose_embedding_dim
from nds_toolbox.utils.helper import compare_decoding_performance

# %%

total_cores = os.cpu_count() or 1
inner_n_jobs = 10

print("total_cores:", total_cores)
print("inner_n_jobs:", inner_n_jobs)

# %%
sim_cond = "waveform_asymmetry_snr"
signal_dir = "../data/simulations"
performance_dir = "../data/performance"

sim_data = np.load(f'{signal_dir}/{sim_cond}_data.npz', allow_pickle=True)
signal_sample = sim_data['signal_sample']
states_sample = sim_data['states_sample']
print("shape of signal_sample", signal_sample.shape)

seed_base = 2026
fs = 250

freq = [20]

rise_ratio = [0.1, 0.9]  # waveform asymmetry will be sampled Uniform distribution between 0.1 to 0.9
rise_ratio_sampling = "uniform"

# Condition 1: burst morphologies
burst_types = ["sine", "asine", "sawtooth"]

# Condition 2: Signal-to-noise ratio (in dB)
snr_range = np.arange(-10, 11, 2)  # varying SNRs from -10 to 10 dB


# %%

def _run_one(sample_id, burst_type_id, snr_id):
    freq_for_emb = float(np.mean(freq))

    num_emb_dpgmm = choose_embedding_dim(
        freq_for_emb,
        fs,
        min_cycles=2.5,
        ensure_odd=True,
    )

    sig = signal_sample[sample_id, burst_type_id, snr_id]
    sig = (sig - np.mean(sig)) / np.std(sig)  # standardize

    st = states_sample[sample_id, burst_type_id, snr_id].astype(int)

    num_states = int(np.ceil(np.log(len(sig))))  # E[K] = a ln n

    # use different seed for each sample
    seed = int(seed_base + sample_id)

    data_info = {
        "sample_id": sample_id,
        "signal": sig,
        "true_states": st,
        "burst_f": freq,
        "fs": fs,
        "burst_type": burst_types[burst_type_id],
        "snr": snr_range[snr_id],
    }

    model_info = {
        "seed": seed,
        "use_dpgmm": True,
        "use_hmm": False,
        "num_states": num_states,

        "num_emb": num_emb_dpgmm,

        "use_model_tqdm": False,
        "use_thresholding": False,
        "filter_freq": np.array(freq),
        "imputing_spurious_states": True,
        "remove_edge_states": True,
        "compute_summary_stats": False,
        "min_samples": int(np.round((2 / np.max(freq)) * fs)),
        "truncate_weights": True,
        "debug_mode": False,
        "n_jobs": inner_n_jobs
    }

    return compare_decoding_performance(data_info=data_info, model_info=model_info, verbose=True)


# %%

"""

# Build the parameter grid: first sample only, all rise ratios
param_iter = itertools.product(
    range(1),                       # Only sample_id = 0
    range(signal_sample.shape[1]),  
    range(signal_sample.shape[2]),
)

total_sims = signal_sample.shape[1]

print("num sims:", total_sims)

from tqdm.auto import tqdm

bar = tqdm(
    param_iter,
    total=total_sims,
    desc="Running sims",
    unit="sim",
)

all_results = [_run_one(i, j) for i, j in bar]

# Save test results separately
os.makedirs(performance_dir, exist_ok=True)

save_path = os.path.join(
    performance_dir,
    f"results_{sim_cond}_test.pkl",
)

with open(save_path, "wb") as f:
    pickle.dump(all_results, f)

print(f"Performance results saved as {save_path}")

"""

# %%


# Build the parameter grid


param_iter = itertools.product(
    range(signal_sample.shape[0]),
    range(signal_sample.shape[1]),
    range(signal_sample.shape[2]),
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
