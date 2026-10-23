
import itertools
import os
import pickle

import numpy as np
from tqdm.auto import tqdm

from nds_toolbox.preprocessing.features import choose_embedding_dim
from nds_toolbox.utils.helper import compare_decoding_performance


# ============================================================
# Parallelization
# ============================================================
total_cores = os.cpu_count() or 1
inner_n_jobs = 10

print("total_cores:", total_cores)
print("inner_n_jobs:", inner_n_jobs)


# ============================================================
# Load simulated data
# ============================================================
sim_cond = "multiple_exponents"

signal_dir = "../data/simulations"
performance_dir = "../data/performance"

data_path = f"{signal_dir}/{sim_cond}_data.npz"

with np.load(data_path, allow_pickle=False) as sim_data:
    signal_sample = sim_data["signal_sample"]
    states_sample = sim_data["states_sample"]

    saved_exponent_range = (
        sim_data["exponent_range"]
        if "exponent_range" in sim_data.files
        else None
    )

assert signal_sample.shape == states_sample.shape

print("shape of signal_sample:", signal_sample.shape)


# ============================================================
# Simulation parameters
# ============================================================
seed_base = 2026
fs = 250

freq = [20]
snr = 2

base_exponent = 1.0

variability_levels_all = np.round(
    np.arange(-1.0, 1.1, 0.2),
    1,
)

expected_exponent_range = np.array([
    (base_exponent, np.round(base_exponent + delta, 1))
    for delta in variability_levels_all
])

# Prefer the condition metadata stored with the simulation.
if saved_exponent_range is not None:

    exponent_range = np.asarray(
        saved_exponent_range,
        dtype=float,
    )

else:
    # Support files generated with or without (1.0, 1.0).
    expected_filtered = expected_exponent_range[
        ~np.isclose(
            expected_exponent_range[:, 0],
            expected_exponent_range[:, 1],
        )
    ]

    n_conditions = signal_sample.shape[1]

    if n_conditions == len(expected_exponent_range):
        exponent_range = expected_exponent_range

    elif n_conditions == len(expected_filtered):
        exponent_range = expected_filtered

    else:
        raise ValueError(
            f"Cannot match {n_conditions} saved conditions "
            "to the expected exponent pairs."
        )


assert signal_sample.shape[1] == len(exponent_range)

# Compute variability directly from each exponent pair.
variability_levels = (
    exponent_range[:, 1] - exponent_range[:, 0]
)

# Exclude conditions where both exponents are identical.
# Preserve original indices in signal_sample.
valid_condition_ids = [
    condition_id
    for condition_id, exponents in enumerate(exponent_range)
    if not np.isclose(exponents[0], exponents[1])
]

print("Exponent pairs:")
for condition_id in valid_condition_ids:
    print(
        condition_id,
        tuple(exponent_range[condition_id]),
        "delta:",
        round(float(variability_levels[condition_id]), 1),
    )

print("Number of valid conditions:", len(valid_condition_ids))


# ============================================================
# Model evaluation
# ============================================================
def _run_one(sample_id, condition_id):

    freq_for_emb = float(np.mean(freq))

    num_emb_dpgmm = choose_embedding_dim(
        freq_for_emb,
        fs,
        min_cycles=2.5,
        ensure_odd=True,
    )

    # Signal
    sig = signal_sample[sample_id, condition_id]

    sig_std = np.std(sig)
    if sig_std == 0 or not np.isfinite(sig_std):
        raise ValueError(
            f"Invalid signal standard deviation: "
            f"sample={sample_id}, condition={condition_id}"
        )

    sig = (sig - np.mean(sig)) / sig_std

    # Ground-truth states
    st = states_sample[
        sample_id,
        condition_id,
    ].astype(int)

    # Truncation level
    num_states = int(np.ceil(np.log(len(sig))))

    # Different seed for each sample
    seed = int(seed_base + sample_id)

    exponents = tuple(exponent_range[condition_id])

    data_info = {
        "sample_id": sample_id,
        "signal": sig,
        "true_states": st,
        "burst_f": freq,
        "fs": fs,
        "snr": snr,
        "exponent_range": exponents,
        "base_exponent": base_exponent,
        "variability_levels": float(
            variability_levels[condition_id]
        ),
    }

    print(
        f"sample_id: {sample_id}, "
        f"exponent_range: {exponents}"
    )

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
        verbose=True,
    )


# ============================================================
# Test run
# ============================================================
"""
param_iter = itertools.product(
    range(1),
    valid_condition_ids,
)

total_sims = len(valid_condition_ids)

bar = tqdm(
    param_iter,
    total=total_sims,
    desc="Running sims",
    unit="sim",
)

all_results = [_run_one(i, j) for i, j in bar]

os.makedirs(performance_dir, exist_ok=True)

save_path = os.path.join(
    performance_dir,
    f"results_{sim_cond}_test.pkl",
)

with open(save_path, "wb") as f:
    pickle.dump(all_results, f)

print(f"Performance results saved as {save_path}")
"""


# ============================================================
# Full evaluation
# ============================================================
param_iter = itertools.product(
    range(signal_sample.shape[0]),
    valid_condition_ids,
)

total_sims = (
    signal_sample.shape[0] * len(valid_condition_ids)
)

print("num sims:", total_sims)

bar = tqdm(
    param_iter,
    total=total_sims,
    desc="Running sims",
    unit="sim",
)

all_results = [_run_one(i, j) for i, j in bar]


# ============================================================
# Save performance results
# ============================================================
os.makedirs(performance_dir, exist_ok=True)

save_path = os.path.join(
    performance_dir,
    f"results_{sim_cond}.pkl",
)

with open(save_path, "wb") as f:
    pickle.dump(all_results, f)

print(f"Performance results saved as {save_path}")
