import argparse
import itertools
import os
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from tqdm.auto import tqdm

from nds_toolbox.preprocessing.features import choose_embedding_dim
from nds_toolbox.utils.helper import compare_decoding_performance


SIMULATION_PATH = Path("../data/simulations/3states_data.npz")
PERFORMANCE_DIR = Path("../data/performance")
RESULT_NAME = "results_3states_hmm_corrected"
FREQUENCY_PAIRS = [(10, 40), (15, 40), (20, 40), (25, 40), (30, 40), (35, 40)]
SNRS = tuple(range(-10, 11, 2))
FS = 250


def run_condition(sample_id, freq, snr, signal, states, n_jobs):
    num_emb = choose_embedding_dim(float(np.mean(freq)), FS, min_cycles=2.5, ensure_odd=True)
    data_info = {
        "sample_id": sample_id,
        "signal": (signal - np.mean(signal)) / np.std(signal),
        "true_states": states.astype(int),
        "burst_f": list(freq),
        "fs": FS,
        "snr": snr,
    }
    model_info = {
        "seed": 2026 + sample_id,
        "use_dpgmm": False,
        "use_hmm": True,
        "num_states": 3,
        "num_models": 10,
        "num_emb": num_emb,
        "use_model_tqdm": False,
        "use_thresholding": False,
        "filter_freq": np.array(freq),
        "imputing_spurious_states": True,
        "compute_summary_stats": False,
        "min_samples": max(1, int(round(0.07 * FS))),
        "truncate_weights": False,
        "debug_mode": False,
        "n_jobs": n_jobs,
    }
    return compare_decoding_performance(
        data_info=data_info, model_info=model_info,
        metrics=("mcc", "ari", "kappa"), verbose=True,
    )


def save_pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temp_dir:
        temp_path = Path(temp_dir) / path.name
        with temp_path.open("wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temp_path.replace(path)


def sample_path(parts_dir, sample_id):
    return parts_dir / f"sample_{sample_id:03d}.pkl"


def fit_samples(parts_dir, selected_sample):
    total_cores = os.cpu_count() or 1
    n_jobs = min(int(os.environ.get("N_PROCS", "10")),
                 int(os.environ.get("SLURM_CPUS_PER_TASK", total_cores)))
    with np.load(SIMULATION_PATH) as data:
        signals = data["signal_sample"]
        states = data["states_sample"]
    if (signals.ndim != 4 or signals.shape != states.shape
            or signals.shape[1:3] != (len(FREQUENCY_PAIRS), len(SNRS))):
        raise ValueError(f"Unexpected simulation grid: signals {signals.shape}, states {states.shape}")

    num_samples = signals.shape[0]
    if selected_sample is not None and not 0 <= selected_sample < num_samples:
        raise ValueError(f"Sample ID must be between 0 and {num_samples - 1}")
    sample_ids = range(num_samples) if selected_sample is None else [selected_sample]
    print(f"Running {len(sample_ids)} sample(s) with {n_jobs} inner workers")
    for sample_id in sample_ids:
        groups = []
        grid = itertools.product(enumerate(FREQUENCY_PAIRS), enumerate(SNRS))
        for (j, freq), (k, snr) in tqdm(grid, total=len(FREQUENCY_PAIRS) * len(SNRS),
                                         desc=f"Sample {sample_id}", unit="condition"):
            groups.append(run_condition(sample_id, freq, snr,
                                        signals[sample_id, j, k], states[sample_id, j, k], n_jobs))
        path = sample_path(parts_dir, sample_id)
        save_pickle(path, {"sample_id": sample_id, "num_samples": num_samples, "groups": groups})
        print(f"Saved {path}", flush=True)


def merge_samples(parts_dir):
    def load_sample(sample_id):
        path = sample_path(parts_dir, sample_id)
        if not path.is_file():
            raise FileNotFoundError(f"Missing sample {sample_id}: {path}")
        with path.open("rb") as stream:
            return pickle.load(stream)

    first = load_sample(0)
    num_samples = first["num_samples"]
    if not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError("Invalid sample count in sample 0")

    results = []
    conditions = list(itertools.product(FREQUENCY_PAIRS, SNRS))
    expected_methods = ("HMM", "HMM (Imputed)")
    for sample_id in range(num_samples):
        part = first if sample_id == 0 else load_sample(sample_id)
        groups = part["groups"]
        if (part["sample_id"] != sample_id or part["num_samples"] != num_samples
                or len(groups) != len(conditions)):
            raise ValueError(f"Incorrect sample metadata or condition count for sample {sample_id}")
        for group, (freq, snr) in zip(groups, conditions):
            if (tuple(row["method"] for row in group) != expected_methods or
                    any(row["sample_id"] != sample_id or tuple(row["burst_f"]) != freq
                        or row["snr"] != snr for row in group)):
                raise ValueError(f"Unexpected result for sample {sample_id}, {freq} Hz, SNR {snr}")
        results.extend(groups)

    path = PERFORMANCE_DIR / f"{RESULT_NAME}.pkl"
    save_pickle(path, results)
    print(f"Saved {len(results)} conditions to {path}")


def main():
    parser = argparse.ArgumentParser(description="Fit and merge the three-state HMM benchmark.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sample-id", type=int, help="Fit one zero-based sample across all conditions.")
    mode.add_argument("--merge", action="store_true", help="Merge completed sample files without fitting.")
    parser.add_argument("--run-id", type=int, help="Keep parts from different Slurm arrays separate.")
    args = parser.parse_args()
    parts_dir = PERFORMANCE_DIR / f"{RESULT_NAME}_parts"
    if args.run_id is not None:
        parts_dir = parts_dir / str(args.run_id)
    if not args.merge:
        fit_samples(parts_dir, args.sample_id)
    if args.sample_id is None:
        merge_samples(parts_dir)


if __name__ == "__main__":
    main()
