
# %% Imports and settings
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize, TwoSlopeNorm
from sklearn.metrics import adjusted_rand_score
from tqdm.auto import tqdm

# Find the project root in both script and interactive execution.
start_dir = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd().resolve()
)
for candidate in (start_dir, *start_dir.parents):
    if (candidate / "nds_toolbox").is_dir() and (candidate / "data").is_dir():
        project_dir = candidate
        break
else:
    raise FileNotFoundError(f"Could not find the TDE-DP-GMM project root from {start_dir}")

sys.path.insert(0, str(project_dir))

from nds_toolbox.models.dpgmm_numpyro import get_states, truncate
from nds_toolbox.utils.helper import prepare_tde_data

sim_cond = "3states"

fig_dir = project_dir / "figures" / sim_cond
fig_dir.mkdir(parents=True, exist_ok=True)

plot_f = (20, 40)
cv_range = np.arange(0.01, 1, 0.02)
reference_cv = 0.1
true_states = 3

FIG_WIDTH = 12
FONT_SIZE = 11
ANNOT_FONT_SIZE = 7

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
def sensitivity_analysis_cv(row, thresholds):
    """Return one result record per threshold, embedding the signal once."""
    signal = np.asarray(row["signal"])
    truth = np.asarray(row["true_states"], dtype=int)
    num_emb = int(row["num_emb"])
    params = row["est_params"]

    tde_signal, _, trimmed_states = prepare_tde_data(signal, truth, num_emb)
    trimmed_states = np.asarray(trimmed_states, dtype=int)

    metadata = {
        "source_row": row.name,  # Unique index in df_original.
        "method": row["method"],
        "burst_f": row["burst_f"],
        "snr": row["snr"],
        "num_emb": num_emb,
        "num_upper_states": len(params["untrunc_weights"]),
    }
    records = []
    for cv in thresholds:
        means, covs, weights = truncate(
            params["untrunc_means"],
            params["untrunc_covs"],
            params["untrunc_weights"],
            mass_threshold=0.99,
            remove_edge_states=True,
            edge_states_cv=cv,
            return_info=False,
        )
        est_states = np.asarray(
            get_states(tde_signal, means, covs, weights), dtype=int
        )

        # ARI does not require matching state labels.
        records.append({
            **metadata,
            "cv_threshold": float(cv),
            "ari": adjusted_rand_score(trimmed_states, est_states),
            "num_active_states": int(np.unique(est_states).size),
        })

    return records


# %% Load source data and select the frequency pair across all SNRs
file_path = project_dir / "data" / "performance" / f"results_{sim_cond}.pkl"
with file_path.open("rb") as f:
    res = pickle.load(f)

df_original = pd.concat([pd.DataFrame(result) for result in res], ignore_index=True)
df_original["burst_f"] = df_original["burst_f"].apply(tuple)
selected_df = df_original.query("burst_f == @plot_f and method == 'DP-GMM'").copy()
if selected_df.empty:
    raise ValueError(f"No DP-GMM results match burst_f={plot_f}.")
selected_df["snr"] = pd.to_numeric(selected_df["snr"], errors="raise")
if not np.isfinite(selected_df["snr"].to_numpy(dtype=float)).all():
    raise ValueError("Selected results contain missing or non-finite SNRs.")


# %% Store individual scores in df: one row per sample and CV threshold
records = []
for _, row in tqdm(selected_df.iterrows(), total=len(selected_df), desc="CV sensitivity"):
    records.extend(sensitivity_analysis_cv(row, cv_range))

df = pd.DataFrame.from_records(records)
df.attrs.update(source_file=str(file_path), sim_cond=sim_cond)
if not np.isfinite(df[["ari", "num_active_states"]].to_numpy()).all():
    raise ValueError("Non-finite scores found; inspect df before taking medians.")


# %% Summarize df (rerun from here to replot without recomputing scores)
snr_order = np.sort(df["snr"].unique())[::-1]  # Seaborn draws the first row at the top.
cv_order = np.sort(df["cv_threshold"].unique())
num_upper_states = int(df["num_upper_states"].max())


def median_heatmap(column):
    return df.pivot_table(
        index="snr", columns="cv_threshold", values=column,
        aggfunc="median", observed=True,
    ).reindex(index=snr_order, columns=cv_order)


median_ari = median_heatmap("ari")
median_active_states = median_heatmap("num_active_states")


# %% Plot annotated Seaborn heatmaps
ari_min = min(0.0, float(median_ari.min().min()))
metric_norm = Normalize(vmin=ari_min, vmax=1.0)
state_norm = TwoSlopeNorm(vmin=1, vcenter=true_states, vmax=num_upper_states)

fig, axes = plt.subplots(
    2, 1, figsize=(FIG_WIDTH + 1, 6), sharex=True, sharey=True,
    constrained_layout=True,
)
common = dict(
    annot=True, annot_kws={"size": ANNOT_FONT_SIZE},
    xticklabels=False, yticklabels=[f"{snr:g}" for snr in snr_order],
    rasterized=True,
)
panels = [
    (median_ari, "Median ARI", "Median ARI", "viridis", metric_norm, ".2f"),
    (median_active_states, "Median active states", "Median active states", "RdBu_r", state_norm, "g"),
]

# Convert CV values to Seaborn's cell positions on the uniform CV grid.
cv_step = cv_order[1] - cv_order[0]
cv_ticks = np.arange(0, 1.01, 0.1)
tick_positions = (cv_ticks - cv_order[0]) / cv_step + 0.5
reference_x = (reference_cv - cv_order[0]) / cv_step + 0.5
for ax, (matrix, title, label, cmap, norm, fmt) in zip(axes, panels):
    sns.heatmap(
        matrix,
        ax=ax, cmap=cmap, norm=norm, fmt=fmt,
        cbar_kws={"label": label, "pad": 0.005}, **common,
    )
    cbar = ax.collections[0].colorbar
    cbar.set_label(label, rotation=270, labelpad=12)
    ax.set(title=None, xlabel="CV threshold", ylabel="SNR (dB)")
    ax.set_xticks(tick_positions, labels=[f"{cv:.1f}" for cv in cv_ticks])
    ax.set_xlim(0, len(cv_order))
    ax.tick_params(axis="both", labelrotation=0, labelbottom=True)
    ax.axvline(reference_x, color="tab:red", linestyle="--", linewidth=2)

state_cbar = axes[1].collections[0].colorbar
state_cbar.set_ticks([1, true_states, num_upper_states])
state_cbar.set_ticklabels([
    "1 (too few)", f"{true_states:g} (target)", f"{num_upper_states:g} (too many)",
])


for fmt in ("svg", "pdf"):
    fig.savefig(
        fig_dir / f"{sim_cond}_sensitivity_cv_threshold_heatmaps.{fmt}",
        bbox_inches="tight",
        pad_inches=0.03,
    )


plt.show()
