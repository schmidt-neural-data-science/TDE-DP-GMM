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
sim_cond = "prior_sensitivity"
signal_dir = "../data/simulations"
performance_dir = "../data/performance"

sim_data = np.load(f'{signal_dir}/{sim_cond}_data.npz', allow_pickle=True)
signal_sample = sim_data['signal_sample']
states_sample = sim_data['states_sample']
print("shape of signal_sample", signal_sample.shape)

seed_base = 2026
fs = 250

freq = [20]

snr_db = 2

# %%
# Condition

prior_range = [0.1, 0.3, 0.5, 3.0, 5.0, 10.0]

alpha_prior_scale_range = prior_range
sigma_prior_scale_range = prior_range
lkj_concentration_range = prior_range

prior_configs = [("baseline", 1.0)]

prior_configs += [
    ("alpha_prior_scale", v) for v in alpha_prior_scale_range
]
prior_configs += [
    ("sigma_prior_scale", v) for v in sigma_prior_scale_range
]
prior_configs += [
    ("lkj_concentration", v) for v in lkj_concentration_range
]

print("prior_configs:", prior_configs)


# %%


def _run_one(sample_id, prior_id):
    freq_for_emb = float(np.mean(freq))

    num_emb_dpgmm = choose_embedding_dim(
        freq_for_emb,
        fs,
        min_cycles=2.5,
        ensure_odd=True,
    )

    sig = signal_sample[sample_id]
    sig = (sig - np.mean(sig)) / np.std(sig)  # standardize

    st = states_sample[sample_id].astype(int)

    num_states = int(np.ceil(np.log(len(sig))))

    # Use different seed for each sample
    seed = int(seed_base + sample_id)

    # Select prior configuration
    prior_name, prior_value = prior_configs[prior_id]

    data_info = {
        "sample_id": sample_id,
        "signal": sig,
        "true_states": st,
        "burst_f": freq,
        "fs": fs,
        "snr": snr_db,
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
        "n_jobs": inner_n_jobs,

        # Default priors
        "alpha_prior_scale": 1.0,
        "sigma_prior_scale": 1.0,
        "lkj_concentration": 1.0,
    }

    # Vary one prior hyperparameter at a time
    if prior_name != "baseline":
        model_info[prior_name] = prior_value

    records = compare_decoding_performance(
        data_info=data_info,
        model_info=model_info,
        verbose=True,
    )

    # Add prior configuration to each record
    for record in records:
        record.update({
            "sample_id": sample_id,
            "prior_id": prior_id,
            "prior_name": prior_name,
            "prior_value": prior_value,
            "alpha_prior_scale": model_info.get("alpha_prior_scale"),
            "sigma_prior_scale": model_info.get("sigma_prior_scale"),
            "lkj_concentration": model_info.get("lkj_concentration"),
        })

    return records


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

run_test = False  # True: first sample only; False: all samples

num_samples = 1 if run_test else signal_sample.shape[0]

total_sims = num_samples * len(prior_configs)

param_iter = itertools.product(
    range(num_samples),
    range(len(prior_configs)),
)

print("num samples:", num_samples)
print("num prior configs:", len(prior_configs))
print("total combinations:", total_sims)

from tqdm.auto import tqdm

bar = tqdm(
    param_iter,
    total=total_sims,
    desc="Running sims",
    unit="sim",
)

all_results = [_run_one(i, j) for i, j in bar]

# %%
# Save results

os.makedirs(performance_dir, exist_ok=True)

suffix = "_test" if run_test else ""

save_path = os.path.join(
    performance_dir,
    f"results_{sim_cond}{suffix}.pkl",
)

with open(save_path, "wb") as f:
    pickle.dump(all_results, f)

print(f"Performance results saved as {save_path}")
