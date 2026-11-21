import itertools
import os
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from tqdm.auto import tqdm

from nds_toolbox.preprocessing.features import choose_embedding_dim
from nds_toolbox.utils.helper import compare_decoding_performance


#%% CPU settings

total_cores = os.cpu_count() or 1
inner_n_jobs = min(
    int(os.environ.get("N_PROCS", "10")),
    int(os.environ.get("SLURM_CPUS_PER_TASK", total_cores)),
)
print("total_cores:", total_cores)
print("inner_n_jobs:", inner_n_jobs)


#%% Simulation settings

sim_cond = "3states"
seed_base = 2026
fs = 250
num_states_hmm = 3

freqs_dist_range = [(10, 40), (15, 40), (20, 40), (25, 40), (30, 40), (35, 40)]
snr_range = np.arange(-10, 11, 2)

imputation_window_sec_dpgmm = 0.03
imputation_window_sec_hmm = 0.07
imputation_window_sec_thresholding = 0.03
min_samples_dpgmm = max(1, round(imputation_window_sec_dpgmm * fs))
min_samples_hmm = max(1, round(imputation_window_sec_hmm * fs))
min_samples_thresholding = max(1, round(imputation_window_sec_thresholding * fs))

print("min_samples (dpgmm):", min_samples_dpgmm)
print("min_samples (hmm):", min_samples_hmm)
print("min_samples (thresholding):", min_samples_thresholding)


#%% Load data

if __name__ == "__main__":
    start_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for project_dir in (start_dir, *start_dir.parents):
        if (project_dir / "nds_toolbox").is_dir():
            break
    else:
        raise FileNotFoundError(f"Could not find the project root from {start_dir}")

    signal_dir = project_dir / "data" / "simulations"
    performance_dir = project_dir / "data" / "performance"
    with np.load(signal_dir / f"{sim_cond}_data.npz") as sim_data:
        signal_sample = sim_data["signal_sample"]
        states_sample = sim_data["states_sample"]

    print("shape of signal_sample", signal_sample.shape)
    if (signal_sample.ndim != 4 or signal_sample.shape != states_sample.shape
            or signal_sample.shape[1:3] != (len(freqs_dist_range), len(snr_range))):
        raise ValueError(f"Unexpected simulation grid: {signal_sample.shape}, {states_sample.shape}")


#%% Run one condition

def _run_one(sample_id, freq_id, snr_id):
    freq = freqs_dist_range[freq_id]
    freq_for_emb = float(np.mean(freq))
    num_emb = choose_embedding_dim(freq_for_emb, fs, min_cycles=2.5, ensure_odd=True)

    snr = snr_range[snr_id]
    sig = signal_sample[sample_id, freq_id, snr_id]
    sig = (sig - np.mean(sig)) / np.std(sig)
    st = states_sample[sample_id, freq_id, snr_id].astype(int)
    seed = int(seed_base + sample_id)

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
        "num_emb": num_emb,
        "num_models": 10,
        "use_model_tqdm": False,
        "imputing_spurious_states": True,
        "compute_summary_stats": False,
        "debug_mode": False,
        "n_jobs": inner_n_jobs,
    }

    # Each call takes one state count and one imputation cutoff.
    # The order below matches the method order in results_3states.pkl.
    model_settings = [
        {
            "use_hmm": True,
            "num_states": num_states_hmm,
            "min_samples": min_samples_hmm,
        },
        {
            "use_thresholding": True,
            "filter_freq": np.array(freq),
            "num_states": len(freq) + 1,
            "min_samples": min_samples_thresholding,
        },
        {
            "use_dpgmm": True,
            "num_states": int(np.ceil(np.log(sig.size))),
            "min_samples": min_samples_dpgmm,
            "truncate_weights": True,
            "remove_edge_states": True,
        },
    ]
    results = []
    for settings in model_settings:
        results.extend(compare_decoding_performance(
            data_info=data_info,
            model_info={**model_info, **settings},
            metrics=("mcc", "ari", "kappa"),
            verbose=True,
        ))
    return results


#%% Build the parameter grid and run simulations

if __name__ == "__main__":
    param_iter = itertools.product(
        range(signal_sample.shape[0]),
        range(signal_sample.shape[1]),
        range(signal_sample.shape[2]),
    )
    total_sims = int(np.prod(signal_sample.shape[:3]))
    print("num sims:", total_sims)
    bar = tqdm(param_iter, total=total_sims, desc="Running sims", unit="sim")
    all_results = [_run_one(i, j, k) for i, j, k in bar]


#%% Save

if __name__ == "__main__":
    performance_dir.mkdir(parents=True, exist_ok=True)
    save_path = performance_dir / f"results_{sim_cond}.pkl"
    with TemporaryDirectory(dir=performance_dir) as temp_dir:
        temp_path = Path(temp_dir) / save_path.name
        with temp_path.open("wb") as f:
            pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        temp_path.replace(save_path)
    print(f"Performance results saved as {save_path}")
