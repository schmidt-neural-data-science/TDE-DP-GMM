import itertools
import os
import pickle

import numpy as np
from tqdm.auto import tqdm

from nds_toolbox.preprocessing.features import choose_embedding_dim
from nds_toolbox.utils.helper import compare_decoding_performance


# %% Execution settings

TEST_MODE = True

total_cores = os.cpu_count() or 1
inner_n_jobs = 10

print("total_cores:", total_cores)
print("inner_n_jobs:", inner_n_jobs)
print("test_mode:", TEST_MODE)


# %% Load simulation data

sim_cond = "waveform_asymmetry_snr"
signal_dir = "../data/simulations"
performance_dir = "../data/performance"

sim_data = np.load(
    os.path.join(signal_dir, f"{sim_cond}_data.npz"),
    allow_pickle=True,
)
signal_sample = sim_data["signal_sample"]
states_sample = sim_data["states_sample"]

# Array layout: (sample, burst_type, SNR, time)
print("shape of signal_sample:", signal_sample.shape)

seed_base = 2026
fs = 250

freq = [20]

# Waveform asymmetry sampled uniformly between 0.1 and 0.9
rise_ratio = [0.1, 0.9]
rise_ratio_sampling = "uniform"

# Condition 1: burst morphology
burst_types = ["sine", "asine"]

# Condition 2: signal-to-noise ratio (dB)
snr_range = np.arange(-10, 11, 2)


imputation_window_sec_dpgmm = 0.03
min_samples_dpgmm = max(1, int(round(imputation_window_sec_dpgmm * fs)))

print("min_samples:dpgmm ", min_samples_dpgmm)
print("min_samples (sec):dpgmm ", min_samples_dpgmm / fs)



# %% Run one condition

def _run_one(sample_id, burst_type_id, snr_id):
    freq_for_emb = float(np.mean(freq))

    num_emb_dpgmm = choose_embedding_dim(
        freq_for_emb,
        fs,
        min_cycles=2.5,
        ensure_odd=True,
    )

    sig = signal_sample[sample_id, burst_type_id, snr_id]
    sig = (sig - np.mean(sig)) / np.std(sig)

    st = states_sample[sample_id, burst_type_id, snr_id].astype(int)

    num_states = int(np.ceil(np.log(len(sig))))

    # Use a different seed for each sample
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
        "min_samples": min_samples_dpgmm,
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


# %% Build the parameter grid

if TEST_MODE:
    # First sample, sawtooth only, all SNRs:
    # signal_sample[0, 2, snr_id]
    sample_ids = [0]
    burst_type_ids = [burst_types.index("sawtooth")]
else:
    # All samples, all burst types, all SNRs
    sample_ids = range(signal_sample.shape[0])
    burst_type_ids = range(signal_sample.shape[1])

snr_ids = range(signal_sample.shape[2])

param_iter = itertools.product(
    sample_ids,
    burst_type_ids,
    snr_ids,
)

total_sims = len(sample_ids) * len(burst_type_ids) * len(snr_ids)

print("num sims:", total_sims)


# %% Run simulations and display the current condition

all_results = []

with tqdm(
    param_iter,
    total=total_sims,
    desc="Testing" if TEST_MODE else "Running sims",
    unit="sim",
) as bar:
    for sample_id, burst_type_id, snr_id in bar:
        bar.set_postfix(
            sample=sample_id,
            burst=burst_types[burst_type_id],
            SNR=f"{snr_range[snr_id]:+g} dB",
        )

        result = _run_one(sample_id, burst_type_id, snr_id)
        all_results.append(result)


# %% Save results

os.makedirs(performance_dir, exist_ok=True)

suffix = "_test" if TEST_MODE else ""
save_path = os.path.join(
    performance_dir,
    f"results_{sim_cond}{suffix}.pkl",
)

with open(save_path, "wb") as f:
    pickle.dump(all_results, f)

print(f"Performance results saved as {save_path}")