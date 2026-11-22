import argparse
import itertools
import os
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from tqdm.auto import tqdm

from nds_toolbox.utils.helper import compare_decoding_performance


PROJECT_DIR = next(p for p in Path(__file__).resolve().parents if (p / "nds_toolbox").is_dir())
SIMULATION_PATH = PROJECT_DIR / "data/simulations/2states_data.npz"
PERFORMANCE_DIR = PROJECT_DIR / "data/performance"
RESULT_NAME = "results_2states"
# Earlier part files used a fixed 20-way split without recording its size.
LEGACY_NUM_PARTS = 20
FREQUENCIES = (10, 20, 30, 40, 50)
SAMPLING_RATES = (100, 250, 500)
EMBEDDINGS = range(3, 52, 4)
METHODS = ("DP-GMM", "HMM")


def condition_grid(num_samples):
    return [
        (sample_id, freq_id, fs_id, num_emb)
        for sample_id, (freq_id, freq), (fs_id, fs), num_emb in itertools.product(
            range(num_samples), enumerate(FREQUENCIES), enumerate(SAMPLING_RATES), EMBEDDINGS
        )
        if (fs, freq) != (500, 50)
    ]


def run_condition(sample_id, freq, fs, num_emb, signal, states, n_jobs):
    data_info = {
        "sample_id": sample_id,
        "signal": (signal - np.mean(signal)) / np.std(signal),
        "true_states": states.astype(int),
        "burst_f": freq,
        "fs": fs,
        "snr": 2,
    }
    model_info = {
        "seed": 2026 + sample_id,
        "use_dpgmm": True,
        "use_hmm": True,
        "num_states": 2,
        "num_emb": num_emb,
        "use_model_tqdm": False,
        "use_thresholding": False,
        "filter_freq": None,
        "imputing_spurious_states": False,
        "compute_summary_stats": False,
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
    print(f"Saved {path}", flush=True)


def part_path(parts_dir, part_id):
    return parts_dir / f"part_{part_id:02d}.pkl"


def fit_conditions(parts_dir, part_id, num_parts):
    with np.load(SIMULATION_PATH, allow_pickle=True) as sim_data:
        signals = sim_data["signal_sample"]
        states = sim_data["states_sample"]
    if (signals.shape != states.shape or signals.shape[1:3] != (len(FREQUENCIES), len(SAMPLING_RATES))
            or signals.shape[0] == 0):
        raise ValueError(f"Unexpected simulation grid: signals {signals.shape}, states {states.shape}")

    num_samples = signals.shape[0]
    conditions = condition_grid(num_samples)
    if part_id is not None:
        conditions = conditions[part_id::num_parts]
    n_jobs = min(int(os.environ.get("N_PROCS", "10")),
                 int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    print(f"Running {len(conditions)} conditions with {n_jobs} inner workers", flush=True)
    groups = []
    for sample_id, j, k, num_emb in tqdm(conditions, desc="Running sims", unit="sim"):
        groups.append(run_condition(sample_id, FREQUENCIES[j], SAMPLING_RATES[k], num_emb,
                                    signals[sample_id, j, k], states[sample_id, j, k], n_jobs))

    if part_id is None:
        save_pickle(PERFORMANCE_DIR / f"{RESULT_NAME}.pkl", groups)
    else:
        save_pickle(part_path(parts_dir, part_id), {
            "num_samples": num_samples, "num_parts": num_parts, "groups": groups,
        })


def merge_parts(parts_dir):
    def load_part(path):
        with path.open("rb") as stream:
            return pickle.load(stream)

    first = load_part(part_path(parts_dir, 0))
    num_parts = first.get("num_parts", LEGACY_NUM_PARTS)
    num_samples = first["num_samples"]
    if not isinstance(num_parts, int) or num_parts < 1 or not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError(f"Invalid part or sample count in {part_path(parts_dir, 0)}")
    conditions = condition_grid(num_samples)
    results = [None] * len(conditions)

    for part_id in range(num_parts):
        path = part_path(parts_dir, part_id)
        part = first if part_id == 0 else load_part(path)
        indices = range(part_id, len(conditions), num_parts)
        groups = part["groups"]
        if (part.get("num_parts", LEGACY_NUM_PARTS) != num_parts
                or part["num_samples"] != num_samples or len(groups) != len(indices)):
            raise ValueError(f"Incorrect part, sample or condition count in {path}")
        for index, group in zip(indices, groups):
            sample_id, j, k, num_emb = conditions[index]
            expected = (sample_id, FREQUENCIES[j], SAMPLING_RATES[k], num_emb)
            if (tuple(row["method"] for row in group) != METHODS or
                    any((row["sample_id"], row["burst_f"], row["fs"], row["num_emb"]) != expected
                        for row in group)):
                raise ValueError(f"Unexpected result for condition {expected} in {path}")
            results[index] = group

    save_pickle(PERFORMANCE_DIR / f"{RESULT_NAME}.pkl", results)
    print(f"Merged {len(results)} conditions from {num_parts} parts")


def main():
    parser = argparse.ArgumentParser(description="Fit and merge the two-state embedding sweep.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--part-id", type=int, help="Fit one zero-based part.")
    mode.add_argument("--merge", action="store_true", help="Merge all completed parts.")
    parser.add_argument("--run-id", type=int, help="Slurm array job ID used to group the parts.")
    parser.add_argument("--num-parts", type=int, default=os.environ.get("SLURM_ARRAY_TASK_COUNT", LEGACY_NUM_PARTS),
                        help="Total parts for fitting. Defaults to the Slurm array size, or 20 outside Slurm.")
    args = parser.parse_args()
    parts_dir = PERFORMANCE_DIR / f"{RESULT_NAME}_parts"
    if args.run_id is not None:
        parts_dir = parts_dir / str(args.run_id)
    if args.merge:
        merge_parts(parts_dir)
    else:
        if args.num_parts < 1 or (args.part_id is not None and not 0 <= args.part_id < args.num_parts):
            parser.error("--num-parts must be positive and --part-id must be in [0, num_parts)")
        fit_conditions(parts_dir, args.part_id, args.num_parts)


if __name__ == "__main__":
    main()
