import itertools
import os
import pickle
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from nds_toolbox.preprocessing.features import choose_embedding_dim
from nds_toolbox.utils.helper import compare_decoding_performance


# %%
# Requires the updated compare_decoding_performance and
# return_performance_info functions supporting the metrics argument.

total_cores = os.cpu_count() or 1
inner_n_jobs = 10

print("total_cores:", total_cores)
print("inner_n_jobs:", inner_n_jobs)


# %%
sim_cond = "sensitivity_upper_k"

project_root = Path(__file__).resolve().parents[1]
data_file = project_root / "data" / "simulations" / f"{sim_cond}_data.npz"
performance_dir = project_root / "data" / "performance"

if not data_file.is_file():
    raise FileNotFoundError(f"Simulation data file not found: {data_file}")

print(f"Loading simulation data: {data_file}", flush=True)

with np.load(data_file, allow_pickle=True) as sim_data:
    signal_sample = sim_data["signal_sample"]
    states_sample = sim_data["states_sample"]

if signal_sample.ndim != 2 or signal_sample.shape != states_sample.shape:
    raise ValueError(
        "signal_sample and states_sample must have the same "
        "(n_samples, n_timepoints) shape."
    )
if 0 in signal_sample.shape:
    raise ValueError("Simulation data must contain at least one nonempty sample.")

print("shape of signal_sample:", signal_sample.shape)

seed_base = 2026
fs = 250
snr = 2
freq = [20]

metrics = ("mcc", "ari", "kappa")

# Vary the upper bound K.
# Practical DP-GMM truncation heuristic.
num_data = signal_sample.shape[-1]

reference_num_states = int(np.ceil(np.log(num_data)))
print("reference_num_states:", reference_num_states)

num_states_vec = [3, 7, 11, 15, 19, 23, 27, 31, 35]

num_emb = choose_embedding_dim(
    float(np.mean(freq)),
    fs,
    min_cycles=2.5,
    ensure_odd=True,
)


# %%
def _run_one(sample_id, cond_id):
    sig = np.asarray(signal_sample[sample_id], dtype=float)
    sig_std = np.std(sig)

    if (
        not np.all(np.isfinite(sig))
        or not np.isfinite(sig_std)
        or sig_std <= 0
    ):
        raise ValueError(f"Invalid signal at sample_id={sample_id}.")

    sig = (sig - np.mean(sig)) / sig_std
    st = np.asarray(states_sample[sample_id], dtype=int)

    seed = int(seed_base + sample_id)

    num_states = num_states_vec[cond_id]

    data_info = {
        "sample_id": sample_id,
        "signal": sig,
        "true_states": st,
        "burst_f": freq,
        "fs": fs,
        "snr": snr,
    }

    model_info = {
        "seed": seed,
        "metrics": metrics,
        "num_emb": num_emb,
        "use_model_tqdm": False,
        "use_thresholding": False,
        "filter_freq": np.array(freq),
        "imputing_spurious_states": False,
        "remove_edge_states": True,
        "compute_summary_stats": False,
        "min_samples": None,
        "truncate_weights": True,
        "debug_mode": False,
        "n_jobs": inner_n_jobs,
    }

    res = compare_decoding_performance(
        data_info=data_info,
        model_info={
            **model_info,
            "use_dpgmm": True,
            "use_hmm": False,
            "num_states": num_states,
            "weight_prior": "dirichlet_process",
        },
        metrics=metrics,
        verbose=True,
    )
    return res


# %%
# Set True to run all K values for the first sample and save separately.
test_mode = False

total_sims = 1 if test_mode else signal_sample.shape[0]
print("num sims:", total_sims)

param_iter = itertools.product(
    range(total_sims),
    range(len(num_states_vec)),
)

all_results = [[] for _ in range(total_sims)]

for sample_id, cond_id in tqdm(
    param_iter,
    total=total_sims * len(num_states_vec),
    desc="Running test" if test_mode else "Running sims",
    unit="fit",
):
    # The helper returns a list of model/variant records for this K.
    all_results[sample_id].extend(_run_one(sample_id, cond_id))


# %%
# Save the existing nested structure:
# all_results[sample_id] contains the records for all K values.
# Each record includes sample_id and num_states (the requested upper bound K).
performance_dir.mkdir(parents=True, exist_ok=True)

suffix = "_test" if test_mode else ""
save_path = performance_dir / f"results_{sim_cond}{suffix}.pkl"

with open(save_path, "wb") as f:
    pickle.dump(all_results, f)

print(f"Performance results saved as {save_path}")
