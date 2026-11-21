# %% Imports and settings
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed, parallel_config
from matplotlib.colors import Normalize
from sklearn.metrics import adjusted_rand_score
from tqdm.auto import tqdm

# Find the project root in both script and interactive execution.
start_dir = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd().resolve()
)
working_dir = Path.cwd().resolve()
for candidate in (start_dir, *start_dir.parents, working_dir, *working_dir.parents):
    if (candidate / "nds_toolbox").is_dir() and (candidate / "data").is_dir():
        project_dir = candidate
        break
else:
    raise FileNotFoundError(f"Could not find the TDE-DP-GMM project root from {start_dir}")

sys.path.insert(0, str(project_dir))
from nds_toolbox.analysis.utils import imputing_mode
from nds_toolbox.preprocessing.features import trim_data

sim_cond = "3states"
methods = ["DP-GMM (Edge-removed)", "HMM"]
plot_f = (20, 40)
fs = 250
knn_windows = np.arange(25) #raw data points (up to 100 ms)

reference_window_ms = {"DP-GMM (Edge-removed)": 30, "HMM": 70}
N_JOBS = -1  # -1: all available cores; 1: sequential.

fig_dir = project_dir / "figures" / sim_cond
fig_dir.mkdir(parents=True, exist_ok=True)
pair_label = "_".join(f"{frequency:g}" for frequency in plot_f) + "Hz"
file_prefix = f"{sim_cond}_sensitivity_knn_imputation_dpgmm_hmm_{pair_label}"

FIG_WIDTH = 11  # Leave space between the 71 two-decimal cell annotations.
FONT_SIZE = 10
ANNOT_FONT_SIZE = 8

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": FONT_SIZE,
    "figure.titlesize": FONT_SIZE,
    "axes.titleweight": "normal",
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.grid": False,
    "figure.autolayout": False,
    "savefig.dpi": 900,
    "figure.dpi": 300,
})


# %% Compute scores for one sample
def sensitivity_analysis_knn_imputation(row, windows):
    """Return one result record per window, always using the original states."""
    original = np.asarray(row["est_states"], dtype=int)
    truth = np.asarray(row["true_states"], dtype=int)
    num_emb = int(row["num_emb"])
    max_frequency = np.max(row["burst_f"])
    if original.ndim != 1 or truth.ndim != 1 or np.any(original == -1):
        raise ValueError(f"Invalid state sequences in source row {row.name}.")
    if truth.shape != original.shape:
        truth = np.asarray(trim_data(truth, num_emb, verbose=False), dtype=int)
    if truth.shape != original.shape or truth.size == 0:
        raise ValueError(f"Cannot align state sequences in source row {row.name}.")

    run_lengths = np.diff(
        np.r_[0, np.flatnonzero(np.diff(original)) + 1, original.size]
    )
    metadata = {
        "source_row": row.name,  # Unique index in df_original.
        "method": row["method"],
        "burst_f": row["burst_f"],
        "snr": row["snr"],
        "num_emb": num_emb,
        "fs": fs,
        "reference_knn_window": reference_window_ms[row["method"]] * fs / 1000,
        "reference_window_ms": reference_window_ms[row["method"]],
    }
    records = []
    for window in map(int, windows):
        short_runs = run_lengths < window
        no_retained_runs = bool(window > 0 and short_runs.all())
        # Window 0 and the toolbox's all-short-runs fallback retain the original.
        states = original.copy()
        if window > 0 and short_runs.any() and not no_retained_runs:
            states = np.asarray(imputing_mode(states, min_samples=window), dtype=int)
        if states.shape != truth.shape or np.any(states == -1):
            raise ValueError(f"Invalid imputation: row {row.name}, window {window}.")

        # ARI does not require matching state labels.
        records.append({
            **metadata,
            "knn_window": window,
            "window_ms": 1000 * window / fs,
            "window_cycles": window * max_frequency / fs,
            "ari": adjusted_rand_score(truth, states),
            "num_changed_samples": int(np.count_nonzero(states != original)),
            "no_retained_runs": no_retained_runs,
        })

    return records


# %% Load source data and select the frequency pair across all SNRs
file_path = project_dir / "data" / "performance" / f"results_{sim_cond}.pkl"
with file_path.open("rb") as f:
    res = pickle.load(f)

df_original = pd.concat([pd.DataFrame(result) for result in res], ignore_index=True)
df_original["burst_f"] = df_original["burst_f"].apply(tuple)
selected_df = df_original.query("burst_f == @plot_f and method in @methods").copy()
missing_methods = set(methods).difference(selected_df["method"].unique())
if missing_methods:
    raise ValueError(f"No results for burst_f={plot_f} and methods: {sorted(missing_methods)}.")
selected_df["snr"] = pd.to_numeric(selected_df["snr"], errors="raise")
if not np.isfinite(selected_df["snr"].to_numpy(dtype=float)).all():
    raise ValueError("Selected results contain missing or non-finite SNRs.")
if "fs" in selected_df and not selected_df["fs"].eq(fs).all():
    raise ValueError("Selected sampling rates differ from fs.")


# %% Store individual scores in df: one row per method, sample and imputation window
sample_columns = ["method", "burst_f", "snr", "num_emb", "true_states", "est_states"]
with parallel_config(backend="loky", inner_max_num_threads=1):
    batches = Parallel(n_jobs=N_JOBS, return_as="generator_unordered")(
        delayed(sensitivity_analysis_knn_imputation)(row, knn_windows)
        for _, row in selected_df[sample_columns].iterrows()
    )
    records = [
        record
        for batch in tqdm(batches, total=len(selected_df), desc="KNN sensitivity")
        for record in batch
    ]

df = pd.DataFrame.from_records(records).sort_values(
    ["source_row", "knn_window"], ignore_index=True
)
df.attrs.update(source_file=str(file_path), sim_cond=sim_cond, plot_f=plot_f, methods=methods)
if not np.isfinite(df["ari"].to_numpy()).all():
    raise ValueError("Non-finite scores found; inspect df before taking medians.")
df.to_pickle(fig_dir / f"{file_prefix}_scores.pkl")


# %% Summarize df (rerun from here to replot without recomputing scores)
# df = pd.read_pickle(fig_dir / f"{file_prefix}_scores.pkl")
snr_order = np.sort(df["snr"].unique())[::-1]  # Seaborn draws the first row at the top.
window_order = np.sort(df["knn_window"].unique())

median_ari = {}
for method in methods:
    method_df = df.loc[df["method"].eq(method)]
    matrix = method_df.pivot_table(
        index="snr", columns="knn_window", values="ari",
        aggfunc="median", observed=True,
    ).reindex(index=snr_order, columns=window_order)
    if matrix.isna().any().any():
        raise ValueError(f"Missing SNR/window combinations for {method}.")
    median_ari[method] = matrix


# %% Plot all three annotated heatmaps in a 3-by-1 layout
metric_norm = Normalize(vmin=0, vmax=1.0)

fig, axes = plt.subplots(
    2, 1, figsize=(FIG_WIDTH, 6), sharex=True, sharey=True,
    constrained_layout=True,
)

# Seaborn uses cell positions; labels convert sample counts to milliseconds.
tick_indices = np.unique(np.r_[np.arange(0, len(window_order), 5), len(window_order) - 1])
tick_labels = [f"{(window_order[i] / fs) * 1000:g}" for i in tick_indices]

for ax, method in zip(axes, methods):
    sns.heatmap(
        median_ari[method],
        ax=ax, cmap="viridis", norm=metric_norm,
        annot=True, fmt=".2f", annot_kws={"size": ANNOT_FONT_SIZE},
        xticklabels=False, yticklabels=[f"{snr:g}" for snr in snr_order],
        cbar=False, rasterized=True,
    )
    ax.set_title("DP-GMM" if method == "DP-GMM (Edge-removed)" else method, loc="left")
    ax.set(xlabel="", ylabel="SNR (dB)")
    ax.set_xticks(tick_indices + 0.5, labels=tick_labels)
    ax.set_xlim(0, len(window_order))
    ax.tick_params(axis="both", labelrotation=0)
    reference_ms = reference_window_ms[method]
    reference_knn_window = reference_ms * fs / 1000
    if window_order[0] <= reference_knn_window <= window_order[-1]:
        reference_x = np.interp(
            reference_knn_window, window_order, np.arange(len(window_order)) + 0.5
        )
        ax.axvline(
            reference_x, color="tab:red", linestyle="--", linewidth=2, alpha=0.5,
            label=f"Default: {reference_ms:g} ms",
        )
        ax.legend(loc="lower right", bbox_to_anchor=(1, 1.01), frameon=False)

axes[-1].set_xlabel("Imputation window (ms; 0 = no imputation)")

# A common normalization and colourbar make all three methods directly comparable.
cbar = fig.colorbar(
    axes[0].collections[0], ax=axes.tolist(), fraction=0.025, pad=0.01,
    format="%.2f",
)
cbar.set_label("Median ARI", rotation=270, labelpad=12)

for fmt in ("svg", "pdf"):
    fig.savefig(
        fig_dir / f"{file_prefix}_heatmaps.{fmt}",
        bbox_inches="tight",
        pad_inches=0.03,
    )

plt.show()
