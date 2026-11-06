import itertools
import os
import pickle

from functools import lru_cache

import numpy as np
from tqdm.auto import tqdm

from nds_toolbox.preprocessing.features import choose_embedding_dim
from nds_toolbox.utils.helper import compare_decoding_performance

# ============================================================
# CPU settings
# ============================================================

total_cores = os.cpu_count() or 1
inner_n_jobs = 2

print("total_cores:", total_cores)
print("inner_n_jobs:", inner_n_jobs)

# ============================================================
# Directories
# ============================================================

sim_cond = "signal_duration"

signal_dir = "../data/simulations"
performance_dir = "../data/performance"

os.makedirs(performance_dir, exist_ok=True)

# ============================================================
# Simulation parameters
# ============================================================

seed_base = 2026
n_samples = 20
fs = 100

freq = [20]
snr_db = 0

burst_type = "sine"
burst_amp_sigma = 0.1

tukey_alpha = 0.25
state_transition = "return_to_baseline"

beta = 1  # Pink noise

# Burst duration in cycles
burst_cycles = [3, 10]

# Noise duration in seconds
noise_duration = [0.2, 2]

# Experimental condition: signal duration
n_seconds_vec = np.array([
    10,  # 10 sec
    30,  # 30 sec
    60,  # 1 min
    180,  # 3 min
    600,  # 10 min
    1800,  # 30 min
    3600,  # 1 hour
    10800,  # 3 hours
])



n_conditions = len(n_seconds_vec)

max_seconds = int(np.max(n_seconds_vec))
max_n_times = int(fs * max_seconds)

print("Maximum signal duration:", max_seconds, "seconds")
print("Maximum number of samples:", max_n_times)

# ============================================================
# Embedding dimension
# ============================================================

freq_for_emb = float(np.mean(freq))

num_emb_dpgmm = choose_embedding_dim(
    freq_for_emb,
    fs,
    min_cycles=2.5,
    ensure_odd=True,
)

print("Embedding dimension:", num_emb_dpgmm)


# ============================================================
# Load simulated signal
# ============================================================

@lru_cache(maxsize=1)
def _load_sample(sample_id):
    data_file = os.path.join(
        signal_dir,
        f"{sim_cond}_sample_{sample_id}.npz",
    )

    with np.load(data_file, allow_pickle=False) as sim_data:
        signal_sample = sim_data["signal_sample"]
        states_sample = sim_data["states_sample"]

    return signal_sample, states_sample


# ============================================================
# Run one simulation
# ============================================================

def _run_one(sample_id, dur_id):
    # --------------------------------------------------------
    # Load the longest signal
    # --------------------------------------------------------

    signal_sample, states_sample = _load_sample(sample_id)

    # --------------------------------------------------------
    # Select recording duration
    # --------------------------------------------------------

    n_seconds = int(n_seconds_vec[dur_id])
    n_times = int(round(n_seconds * fs))

    sig = np.array(
        signal_sample[:n_times],
        dtype=np.float32,
        copy=True,
    )

    st = states_sample[:n_times].astype(int)

    # --------------------------------------------------------
    # Standardize
    # --------------------------------------------------------

    sig_mean = np.mean(sig, dtype=np.float64)
    sig_std = np.std(sig, dtype=np.float64)

    if not np.isfinite(sig_std) or sig_std <= 0:
        raise ValueError(
            f"Invalid standard deviation for "
            f"sample {sample_id}, duration {n_seconds}s."
        )

    sig = (sig - sig_mean) / sig_std

    # --------------------------------------------------------
    # Number of states
    # --------------------------------------------------------

    num_states = int(np.ceil(np.log(len(sig))))

    # --------------------------------------------------------
    # Random seed
    # --------------------------------------------------------

    seed = int(seed_base + sample_id)

    # --------------------------------------------------------
    # Data information
    # --------------------------------------------------------

    data_info = {
        "sample_id": sample_id,
        "signal": sig,
        "true_states": st,
        "burst_f": freq,
        "fs": fs,
        "snr": snr_db,
        "n_seconds": n_seconds_vec[dur_id],
    }

    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

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

        "min_samples": int(
            np.round((2 / np.max(freq)) * fs)
        ),

        "truncate_weights": True,
        "debug_mode": False,
        "n_jobs": inner_n_jobs,
    }

    return compare_decoding_performance(
        data_info=data_info,
        model_info=model_info,
        metrics=("mcc", "ari", "kappa"),
        verbose=True,
    )


"""
# ============================================================
# Test: first sample, all recording durations
# ============================================================

param_iter = itertools.product(
    range(1),
    range(n_conditions),
)

total_sims = n_conditions

print("num sims:", total_sims)

bar = tqdm(
    param_iter,
    total=total_sims,
    desc="Running sims",
    unit="sim",
)

all_results = [_run_one(i, j) for i, j in bar]

# ============================================================
# Save test results
# ============================================================

save_path = os.path.join(
    performance_dir,
    f"results_{sim_cond}_test.pkl",
)

with open(save_path, "wb") as f:
    pickle.dump(all_results, f)

print(f"Performance results saved as {save_path}")

"""

param_iter = itertools.product(
    range(n_samples),
    range(n_conditions),
)

total_sims = n_samples * n_conditions

print("num sims:", total_sims)

bar = tqdm(
    param_iter,
    total=total_sims,
    desc="Running sims",
    unit="sim",
)

all_results = [_run_one(i, j) for i, j in bar]


# Save
save_path = os.path.join(
    performance_dir,
    f"results_{sim_cond}.pkl",
)

with open(save_path, "wb") as f:
    pickle.dump(all_results, f)

print(f"Performance results saved as {save_path}")
