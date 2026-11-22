# %% Settings and imports


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.signal import welch

from nds_toolbox.preprocessing.features import compute_wavelet

# %%


FIG_WIDTH = 7.5
MY_FONT_SIZE = 9

plt.rcParams.update({
    # --- Fonts ---
    "font.family": "Arial",
    "font.size": MY_FONT_SIZE,

    # --- Legend ---
    "legend.fontsize": MY_FONT_SIZE,
    "legend.frameon": False,

    # --- Titles ---
    "figure.titlesize": MY_FONT_SIZE,
    "axes.titleweight": "normal",
    "axes.titlesize": MY_FONT_SIZE,

    # --- Axes ---
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.edgecolor": "black",
    "axes.unicode_minus": True,
    "axes.labelsize": MY_FONT_SIZE,

    # --- Lines & markers ---
    "lines.linewidth": 1.0,
    "lines.markersize": 0.7,

    # --- Ticks ---
    "xtick.labelsize": MY_FONT_SIZE,
    "ytick.labelsize": MY_FONT_SIZE,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.bottom": True,
    "ytick.left": True,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "axes.linewidth": 0.7,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,

    # Grid
    "axes.grid": False,
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.5,

    # --- Layout ---
    "figure.autolayout": True,

    # --- DPI ---
    "savefig.dpi": 600,
    "figure.dpi": 100,
})

# %%
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

from nds_toolbox.preprocessing.features import trim_data

sim_cond = "3states"

fs = 250  # Sampling frequency in Hz.

# Condition 1: frequency distances (3 states)
freqs_dist_range = [[10, 40], [15, 40], [20, 40], [25, 40], [30, 40], [35, 40]]  # 30, 25, 20, 15, 10, 5Hz diff

# Condition 2: Signal-to-noise ratio (in dB)
snr_range = np.arange(-10, 11, 2)  # varying SNRs from -10 to 10 dB

fig_dir = project_dir / "figures" / sim_cond
fig_dir.mkdir(parents=True, exist_ok=True)

# %% load the simulated signals
signal_dir = project_dir / "data" / "simulations"
sim_data = np.load(f'{signal_dir}/{sim_cond}_data.npz')

# load the simulated data
signal_sample = sim_data['signal_sample']
states_sample = sim_data['states_sample']
bursts_sample = sim_data['bursts_sample']

# %% Load all frequency pairs and SNRs

file_path = project_dir / "data" / "performance" / f"results_{sim_cond}.pkl"
df = pd.concat(map(pd.DataFrame, pd.read_pickle(file_path)), ignore_index=True)
df["burst_f"] = df["burst_f"].apply(tuple)

# extract number of active states for dpgmm
df["num_active_states"] = df["est_params"].apply(
    lambda x: x.get("num_active_states", np.nan)
    if isinstance(x, dict) else np.nan
)

# Imputation can remove states from the decoded sequence.
imputed_methods = [
    "DP-GMM (Imputed)",
    "HMM (Imputed)",
    "Thresholding (Imputed)",
    "DP-GMM (Imputed & Edge-removed)",
]

mask = df["method"].isin(imputed_methods)
df.loc[mask, "num_active_states"] = df.loc[mask, "est_states"].apply(
    lambda states: np.unique(states).size
)

#%%
def relabel_freqs(burst_f):
    f0, f1 = burst_f
    label = f"{str(f0)}&{str(f1)}"
    return label


# %%

# Compute PSDs separately for each simulated signal.
psds = []

for sample_id in range(signal_sample.shape[0]):
    for freq_id in range(signal_sample.shape[1]):
        for snr_id in range(signal_sample.shape[2]):
            f, psd = welch(
                signal_sample[sample_id, freq_id, snr_id, :],
                fs=fs,
            )
            psds.append(psd)

psds = np.asarray(psds).reshape(
    signal_sample.shape[0],
    signal_sample.shape[1],
    signal_sample.shape[2],
    -1,
)

# Average across simulated samples.
mean_psds = psds.mean(axis=0)

# Plot the six frequency-pair conditions.
norm = Normalize(
    vmin=snr_range.min(),
    vmax=snr_range.max(),
)
cmap = plt.get_cmap("viridis")

fig, axes = plt.subplots(
    2, 3,
    figsize=(FIG_WIDTH - 1, 4),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)
axes = axes.ravel()

for col_idx, freq in enumerate(freqs_dist_range):
    ax = axes[col_idx]

    for snr_id, snr in enumerate(snr_range):
        ax.plot(
            f,
            mean_psds[col_idx, snr_id, :],
            color=cmap(norm(snr)),
            linewidth=0.7,
            label=f"{snr:g}",
        )

    ax.set_yscale("log")
    ax.set_title(f"{freq[0]} Hz & {freq[1]} Hz")
    ax.set_xlim(1, 60)
    ax.set_xticks(np.arange(10, 61, 10))

    # X-axis labels only on the bottom row.
    if col_idx >= 3:
        ax.set_xlabel("Frequency (Hz)")

    # Y-axis labels only on the left column.
    if col_idx % 3 == 0:
        ax.set_ylabel("Mean PSD (a.u.)")

# Shared SNR legend below the panels.
handles, labels = axes[0].get_legend_handles_labels()

# Arrange ascending SNR values across rows.
n_columns = 6
order = np.argsort(snr_range)
order = np.concatenate([
    order[col::n_columns] for col in range(n_columns)
])

legend = fig.legend(
    [handles[i] for i in order],
    [labels[i] for i in order],
    title="SNR (dB)",
    loc="outside lower center",
    ncol=n_columns,
    frameon=True,
    fontsize=MY_FONT_SIZE,
    title_fontsize=MY_FONT_SIZE,
    handlelength=1.8,
    handletextpad=0.6,
    labelspacing=0.4,
    borderaxespad=0,
)

# Slightly thicker legend lines for readability.
for line in legend.get_lines():
    line.set_linewidth(1.5)

# Save the figure.
file_path = fig_dir / "mean_psd_by_snr.pdf"

fig.savefig(
    file_path,
    bbox_inches="tight",
    pad_inches=0.1,
    transparent=False,
)

print(f"Figure saved to: {file_path.resolve()}")

plt.show()

# %%

num_states = 3  # ground truth
snr_db = 2
snr_id = np.where(snr_range == snr_db)[0][0]
plot_f = (20, 40)
freq_id = np.where((np.array(freqs_dist_range) == np.array(plot_f)).all(axis=1))[0][0]

sample_id = 1

res = df.query("burst_f == @plot_f and snr == @snr_db")
dpgmm_res = res.query("method == 'DP-GMM (Imputed & Edge-removed)'").iloc[sample_id]
num_emb = dpgmm_res.num_emb

signal = signal_sample[sample_id, freq_id, snr_id]
trimmed_signal = trim_data(signal, num_emb)
trimmed_states = trim_data(states_sample[sample_id, freq_id, snr_id], num_emb).astype(int)
trimmed_bursts = trim_data(bursts_sample[sample_id, freq_id, snr_id], num_emb)
states_onehot = np.eye(num_states)[trimmed_states]  # onehot vector for true states
trimmed_time_vec = np.arange(len(trimmed_states)) / fs

true_num_states = np.unique(trimmed_states).size

hmm_res = res.query("method == 'HMM (Imputed)'").iloc[sample_id]
thresh_res = res.query("method == 'Thresholding (Imputed)'").iloc[sample_id]

hmm_states = hmm_res.est_states
hmm_states_onehot = np.eye(num_states)[hmm_states]

dpgmm_states = dpgmm_res.est_states
num_states_dpgmm_upper = dpgmm_res.num_states
dpgmm_states_onehot = np.eye(num_states)[dpgmm_states]

# crop threshold length
thresh_states = trim_data(thresh_res.est_states, num_emb)

thresh_states_onehot = np.eye(num_states)[thresh_states]

# Run after creating df and preparing the signal/state example.
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties

import seaborn as sns

# ---------------------------------------------------------------------
# Metric and methods
# ---------------------------------------------------------------------
metric = "ari"

metric_labels = {
    "mcc": "MCC",
    "ari": "ARI",
    "kappa": "Cohen's kappa",
}

metric = metric.strip().lower()

metric_label = metric_labels[metric]

plot_methods = [
    "Thresholding",
    "Thresholding (Imputed)",
    "HMM",
    "HMM (Imputed)",
    "DP-GMM (Edge-removed)",
    "DP-GMM (Imputed & Edge-removed)",
]

heatmap_method = "DP-GMM (Imputed & Edge-removed)"


# ---------------------------------------------------------------------
# Prepare performance data
# ---------------------------------------------------------------------
def frequency_pair(values):
    # Support either tuples/lists or strings such as "20&40".
    if isinstance(values, str):
        values = values.split("&")
    return tuple(float(value) for value in values)


df_metric = df[
    ["method", "snr", "burst_f", metric]
].copy()

df_metric["burst_f"] = df_metric["burst_f"].map(frequency_pair)
df_metric = df_metric.dropna(subset=[metric])

plot_frequency = frequency_pair(plot_f)

df_selected = df_metric.loc[
    df_metric["burst_f"].map(
        lambda values: values == plot_frequency
    )
]

if df_selected.empty:
    raise ValueError(
        f"No non-missing {metric} scores for frequencies {plot_f}."
    )

required_dpgmm_methods = {
    "DP-GMM (Edge-removed)",
    "DP-GMM (Imputed & Edge-removed)",
}
missing_dpgmm_methods = required_dpgmm_methods.difference(
    df_selected["method"].unique()
)

if missing_dpgmm_methods:
    raise ValueError(
        f"No non-missing {metric} scores for {plot_f} and methods: "
        f"{sorted(missing_dpgmm_methods)}"
    )

# Recompute summaries from raw scores on every run.
df_metric_summary = (
    df_selected.groupby(["method", "snr"], as_index=False)
    .agg(
        metric_n=(metric, "count"),
        metric_median=(metric, "median"),
        metric_q25=(metric, lambda values: values.quantile(0.25)),
        metric_q75=(metric, lambda values: values.quantile(0.75)),
    )
)

dpgmm_metric_heat = (
    df_metric.loc[df_metric["method"].eq(heatmap_method)]
    .pivot_table(
        index="burst_f",
        columns="snr",
        values=metric,
        aggfunc="median",
    )
    .sort_index(axis=0)
    .sort_index(axis=1)
)

if dpgmm_metric_heat.empty:
    raise ValueError(
        f"No non-missing {metric} scores for {heatmap_method}."
    )

dpgmm_metric_heat.index = [
    "&".join(f"{value:g}" for value in pair)
    for pair in dpgmm_metric_heat.index
]

# Preserve the shared heatmap scale across methods.
metric_medians = df_metric.groupby(
    ["method", "burst_f", "snr"]
)[metric].median()

metric_vmin = float(metric_medians.min())
metric_vmax = float(metric_medians.max())

metric_ymin = 0

# ---------------------------------------------------------------------
# Signal example and wavelet
# ---------------------------------------------------------------------
plot_t = 20
n_plot = int(plot_t * fs)

t, f, amp = compute_wavelet(
    trimmed_signal[:n_plot],
    fs,
    f_min=1,
    f_max=80,
    df=0.5,
    wavelet="cmor1.5-1.0",
)

cmap = plt.get_cmap("tab10")
colors = ["#E0E0E0"] + [cmap(i) for i in range(num_states)]
state_labels = [f"S{i + 1}" for i in range(num_states)]

# ---------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------
mosaic = [
    ["a1", "a1", "."],
    ["a2", "a2", "cbar1"],
    ["a3", "a3", "leg"],
    ["a4", "a4", "leg"],
    ["a5", "a5", "leg"],
    ["a6", "a6", "leg"],
    ["b", "c", "cbar2"],
]

fig, axes = plt.subplot_mosaic(
    mosaic,
    figsize=(FIG_WIDTH, 5),
    constrained_layout=True,
    gridspec_kw={
        "height_ratios": [0.7, 0.6, 0.2, 0.2, 0.2, 0.2, 2],
        "width_ratios": [1, 1, 0.04],
    },
)

fig.set_constrained_layout_pads(
    w_pad=0.0,
    h_pad=0.0,
    wspace=0.01,
    hspace=0.001,
)

# ---------------------------------------------------------------------
# Panel A: simulated signal
# ---------------------------------------------------------------------
axes["a1"].plot(
    trimmed_time_vec[:n_plot],
    trimmed_signal[:n_plot],
    lw=0.2,
    color="k",
)

axes["a1"].plot(
    trimmed_time_vec[:n_plot],
    -2 + 1.3 * (
            trimmed_bursts[:n_plot] * (trimmed_states[:n_plot] == 1)
    ),
    lw=0.2,
    color=colors[1],
)

axes["a1"].plot(
    trimmed_time_vec[:n_plot],
    -3 + 1.3 * (
            trimmed_bursts[:n_plot] * (trimmed_states[:n_plot] == 2)
    ),
    lw=0.2,
    color=colors[2],
)

axes["a1"].tick_params(
    axis="both",
    which="both",
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False,
)

axes["a1"].grid(False)
axes["a1"].set_frame_on(False)

# ---------------------------------------------------------------------
# Panel A: wavelet
# ---------------------------------------------------------------------
im = axes["a2"].imshow(
    amp,
    extent=[t[0], t[-1], f[0], f[-1]],
    origin="lower",
    aspect="auto",
    cmap="viridis",
    vmin=amp.min(),
    vmax=amp.max(),
)

axes["a2"].tick_params(
    axis="x",
    which="both",
    bottom=False,
    top=False,
    labelbottom=False,
    labeltop=False,
)

axes["a2"].yaxis.set_major_locator(MultipleLocator(20))
axes["a2"].set_ylim(1, 55)
axes["a2"].set_ylabel("Freq.(Hz)")

cb = fig.colorbar(im, cax=axes["cbar1"])
cb.set_label("Amp.(a.u.)", rotation=270, labelpad=12)
cb.ax.tick_params(which="both", length=2, width=0.4, pad=1)

# ---------------------------------------------------------------------
# Panel A: state time courses
# These use the state arrays prepared in earlier cells.
# ---------------------------------------------------------------------
axes["a3"].stackplot(
    trimmed_time_vec[:n_plot],
    states_onehot[:n_plot].T,
    linewidth=0, step="mid", edgecolor="none",
    colors=colors,
    labels=state_labels,
)
axes["a3"].set_ylabel(
    "True States", rotation=0, labelpad=30, va="center"
)

axes["a4"].stackplot(
    trimmed_time_vec[:n_plot],
    dpgmm_states_onehot[:n_plot].T,
    linewidth=0, step="mid", edgecolor="none",
    colors=colors,
)
axes["a4"].set_ylabel(
    "DP-GMM", rotation=0, labelpad=30, va="center"
)

axes["a5"].stackplot(
    trimmed_time_vec[:n_plot],
    hmm_states_onehot[:n_plot].T,
    linewidth=0, step="mid", edgecolor="none",
    colors=colors,
)
axes["a5"].set_ylabel(
    "HMM", rotation=0, labelpad=30, va="center"
)

axes["a6"].stackplot(
    trimmed_time_vec[:n_plot],
    thresh_states_onehot[:n_plot].T,
    linewidth=0, step="mid", edgecolor="none",
    colors=colors,
)
axes["a6"].set_ylabel(
    "Thresholding", rotation=0, labelpad=30, va="center"
)

for key in ["a3", "a4", "a5"]:
    axes[key].tick_params(
        axis="x",
        which="both",
        bottom=False,
        labelbottom=False,
        length=0,
    )
    axes[key].tick_params(
        axis="y",
        which="both",
        left=False,
        labelleft=False,
    )
    axes[key].set_yticks([])
    axes[key].set_xticks([])
    axes[key].spines["bottom"].set_visible(True)
    axes[key].set_ylim(0, 1)

for key in ["a1", "a2", "a3", "a4", "a5"]:
    axes[key].sharex(axes["a6"])

axes["a6"].set_xlabel("Time (s)")
axes["a6"].xaxis.set_major_locator(MultipleLocator(1.0))
axes["a6"].tick_params(
    axis="y",
    which="both",
    left=False,
    labelleft=False,
)
axes["a6"].set_ylim(0, 1)
axes["a6"].set_xlim(0, plot_t)
axes["a6"].spines["bottom"].set_visible(True)

state_handles, state_legend_labels = (
    axes["a3"].get_legend_handles_labels()
)

axes["leg"].axis("off")

state_legend = axes["leg"].legend(
    state_handles,
    state_legend_labels,
    loc="center left",
    bbox_to_anchor=(-0.55, 0.5),
    bbox_transform=axes["leg"].transAxes,
    frameon=False,
    handlelength=1,
    borderaxespad=0.0,
)

state_legend.set_in_layout(False)

# ---------------------------------------------------------------------
# Panel B: median metric and interquartile range
# ---------------------------------------------------------------------
thresh_color = "#999999"
hmm_color = "#1f77b4"
dpgmm_color = "#d62728"

palette = {
    "Thresholding": thresh_color,
    "Thresholding (Imputed)": thresh_color,
    "HMM": hmm_color,
    "HMM (Imputed)": hmm_color,
    "DP-GMM (Edge-removed)": dpgmm_color,
    "DP-GMM (Imputed & Edge-removed)": dpgmm_color,
}

dashes = {
    "Thresholding": (2, 1),
    "Thresholding (Imputed)": (None, None),
    "HMM": (2, 1),
    "HMM (Imputed)": (None, None),
    "DP-GMM (Edge-removed)": (2, 1),
    "DP-GMM (Imputed & Edge-removed)": (None, None),
}

markers = {
    "Thresholding": "s",
    "Thresholding (Imputed)": "s",
    "HMM": "o",
    "HMM (Imputed)": "o",
    "DP-GMM (Edge-removed)": "D",
    "DP-GMM (Imputed & Edge-removed)": "D",
}

for method in plot_methods:
    df_sub = (
        df_metric_summary.loc[
            df_metric_summary["method"].eq(method)
        ]
        .sort_values("snr")
    )

    if df_sub.empty:
        continue

    linestyle = (
        "-"
        if dashes[method] == (None, None)
        else (0, dashes[method])
    )

    axes["b"].plot(
        df_sub["snr"],
        df_sub["metric_median"],
        label=method,
        color=palette[method],
        marker=markers[method],
        linestyle=linestyle,
        lw=1.2,
        ms=2,
        alpha=0.7,
        clip_on=False,
    )

    axes["b"].fill_between(
        df_sub["snr"],
        df_sub["metric_q25"],
        df_sub["metric_q75"],
        color=palette[method],
        alpha=0.06,
        clip_on=True,
        linewidth=0,
    )

axes["b"].set_xticks(snr_range)
axes["b"].set_xlim(min(snr_range), max(snr_range))
axes["b"].set_ylim(metric_ymin, 1)
axes["b"].yaxis.set_major_locator(MultipleLocator(0.1))
axes["b"].set_xlabel("SNR (dB)")
axes["b"].set_ylabel(metric_label)
axes["b"].set_title(
    f"{plot_frequency[0]:g} Hz & {plot_frequency[1]:g} Hz",
    pad=0.3,
)

handles_left = [
    Line2D(
        [0], [0],
        color=dpgmm_color,
        marker="D",
        lw=1.2,
        ms=4,
        linestyle="-",
        label="DP-GMM",
    ),
    Line2D(
        [0], [0],
        color=hmm_color,
        marker="o",
        lw=1.2,
        ms=4,
        linestyle="-",
        label="HMM",
    ),
    Line2D(
        [0], [0],
        color=thresh_color,
        marker="s",
        lw=1.2,
        ms=4,
        linestyle="-",
        label="Thresholding",
    ),
]

handles_right = [
    Line2D(
        [0], [0],
        color="k",
        lw=1.2,
        linestyle="--",
        label="No imputation",
    ),
]

handles = handles_left + handles_right
labels = [handle.get_label() for handle in handles]

axes["b"].legend(
    handles,
    labels,
    ncols=1,
    loc="lower right",
    frameon=False,
    columnspacing=1.3,
    handlelength=1.3,
    borderaxespad=0.0,
)

# ---------------------------------------------------------------------
# Panel C: DP-GMM (Imputed & Edge-removed) heatmap
# ---------------------------------------------------------------------
sns.heatmap(
    dpgmm_metric_heat,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 6},
    cmap="viridis",
    vmin=metric_vmin,
    vmax=metric_vmax,
    cbar=True,
    cbar_ax=axes["cbar2"],
    ax=axes["c"],
)

cbar = axes["c"].collections[0].colorbar
cbar.set_label(
    f"Median {metric_label}",
    rotation=270,
    labelpad=12,
)
cbar.ax.tick_params(
    which="both",
    length=2,
    width=0.4,
    pad=1,
)
cbar.outline.set_visible(True)
cbar.outline.set_linewidth(0.8)

axes["c"].set_title("DP-GMM")
axes["c"].set_xlabel("SNR (dB)")
axes["c"].set_ylabel("Frequency (Hz)")

# ---------------------------------------------------------------------
# Panel labels
# ---------------------------------------------------------------------
fig.canvas.draw()
fig.set_constrained_layout(False)

panel_font = FontProperties(
    family="Helvetica Neue",
    weight="heavy",
    size=14,
)

label_kw = dict(
    fontproperties=panel_font,
    color="k",
    ha="left",
    va="top",
    path_effects=[
        pe.withStroke(linewidth=1, foreground="0.30")
    ],
)

pos_a = axes["a1"].get_position()
pos_b = axes["b"].get_position()
pos_c = axes["c"].get_position()

x_left = 0.030

fig.text(
    x_left,
    min(pos_a.y1 + 0.035, 0.985),
    "A",
    transform=fig.transFigure,
    **label_kw,
)

fig.text(
    x_left,
    pos_b.y1 + 0.020,
    "B",
    transform=fig.transFigure,
    **label_kw,
)

fig.text(
    pos_c.x0 - 0.09,
    pos_c.y1 + 0.020,
    "C",
    transform=fig.transFigure,
    **label_kw,
)

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------
fig.savefig(
    fig_dir / f"{sim_cond}_results_{metric}.svg",
    format="svg",
    bbox_inches="tight",
    bbox_extra_artists=[state_legend],
    pad_inches=0.02,
)

fig.savefig(
    fig_dir / f"{sim_cond}_results_{metric}.pdf",
    format="pdf",
    bbox_inches="tight",
    bbox_extra_artists=[state_legend],
    pad_inches=0.02,
)

plt.show()

# %%
# heatmaps

import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

dpgmm_heat = (df.query("method == 'DP-GMM (Edge-removed)'")
              .pivot_table(index="burst_f", columns="snr", values=metric, aggfunc="median")
              .sort_index(axis=0)
              .sort_index(axis=1))

hmm_heat = (df.query("method == 'HMM'")
            .pivot_table(index="burst_f", columns="snr", values=metric, aggfunc="median")
            .sort_index(axis=0)  # freq ascending
            .sort_index(axis=1))  # snr ascending

thresh_heat = (df.query("method == 'Thresholding'")
               .pivot_table(index="burst_f", columns="snr", values=metric, aggfunc="median")
               .sort_index(axis=0)
               .sort_index(axis=1))

dpgmm_imputed_heat = (df.query("method == 'DP-GMM (Imputed & Edge-removed)'")
                      .pivot_table(index="burst_f", columns="snr", values=metric, aggfunc="median")
                      .sort_index(axis=0)
                      .sort_index(axis=1))

hmm_imputed_heat = (df.query("method == 'HMM (Imputed)'")
                    .pivot_table(index="burst_f", columns="snr", values=metric, aggfunc="median")
                    .sort_index(axis=0)  # freq ascending
                    .sort_index(axis=1))  # snr ascending

thresh_imputed_heat = (df.query("method == 'Thresholding (Imputed)'")
                       .pivot_table(index="burst_f", columns="snr", values=metric, aggfunc="median")
                       .sort_index(axis=0)
                       .sort_index(axis=1))

mats = [
    dpgmm_heat, hmm_heat, thresh_heat,
    dpgmm_imputed_heat, hmm_imputed_heat, thresh_imputed_heat
]
for matrix in mats:
    matrix.index = matrix.index.map(relabel_freqs)

vmin = np.nanmin([m.to_numpy() for m in mats])
vmax = np.nanmax([m.to_numpy() for m in mats])

common = dict(
    annot=True, fmt=".2f", annot_kws={"size": 5.2},
    cmap="viridis", vmin=vmin, vmax=vmax,
    cbar=False
)

fig, ax = plt.subplots(2, 3, figsize=(FIG_WIDTH, 4), sharex=True, sharey=True)

sns.heatmap(dpgmm_imputed_heat, **common, ax=ax[0, 0])
ax[0, 0].set_title("DP-GMM")

sns.heatmap(hmm_imputed_heat, **common, ax=ax[0, 1])
ax[0, 1].set_title("HMM")

sns.heatmap(thresh_imputed_heat, **common, ax=ax[0, 2])
ax[0, 2].set_title("Thresholding")

sns.heatmap(dpgmm_heat, **common, ax=ax[1, 0])
ax[1, 0].set_title("DP-GMM (No imputation)")

sns.heatmap(hmm_heat, **common, ax=ax[1, 1])
ax[1, 1].set_title("HMM (No imputation)")

sns.heatmap(thresh_heat, **common, ax=ax[1, 2])
ax[1, 2].set_title("Thresholding")

sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap="viridis")
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax.ravel().tolist(), fraction=0.02, pad=-0.05)
cbar.set_label(f"Median {metric_label}", rotation=270, labelpad=12)

for a in ax.flat:
    a.set_xlabel("")
    a.set_ylabel("")

fig.supylabel("Frequency (Hz)", fontsize=MY_FONT_SIZE)

ax[1, 1].set_xlabel("SNR (dB)", fontsize=MY_FONT_SIZE)

plt.tight_layout()

fig.savefig(
    fig_dir / f"{sim_cond}_heatmaps.svg",
    bbox_inches="tight",
    pad_inches=0.03,
)

fig.savefig(
    fig_dir / f"{sim_cond}_heatmaps.pdf",
    bbox_inches="tight",
    pad_inches=0.03,
)

plt.show()

# %% performance before imputation

dpgmm_res = res.query("method == 'DP-GMM (Edge-removed)'").iloc[sample_id]
hmm_res = res.query("method == 'HMM'").iloc[sample_id]
thresh_res = res.query("method == 'Thresholding'").iloc[sample_id]

# Saved scores for the same sample rows used in the example plots.
report_metrics = [
    name for name in ("ari", "mcc", "kappa")
    if name in df.columns and df[name].notna().any()
]
performance_report = pd.DataFrame(
    [res.loc[res["method"].eq(method)].iloc[sample_id][report_metrics]
     for method in plot_methods],
    index=pd.Index(plot_methods, name="method"),
)
print(
    f"\nSaved performance for sample_id={sample_id}, "
    f"frequencies={plot_f[0]} & {plot_f[1]} Hz, SNR={snr_db} dB "
    "(full evaluated sequences):"
)
print(performance_report.to_string(float_format="{:.4f}".format))

# Replicate summaries for this condition, using the same statistics as panel B.
print(f"\nAcross simulation samples at the same condition: {metric_label}, n and quartiles")
print(
    df_metric_summary.loc[
        df_metric_summary["method"].isin(plot_methods)
        & df_metric_summary["snr"].eq(snr_db)
    ].to_string(index=False, float_format="{:.4f}".format)
)

dpgmm_states = dpgmm_res.est_states
hmm_states = hmm_res.est_states
thresh_states = trim_data(thresh_res.est_states, num_emb)

dpgmm_states_onehot = np.eye(num_states)[dpgmm_states]
hmm_states_onehot = np.eye(num_states)[hmm_states]
thresh_states_onehot = np.eye(num_states)[thresh_states]

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plot_t = 20
# compute wavelet
t, f, amp = compute_wavelet(trimmed_signal[:plot_t * fs], fs, f_min=1, f_max=80, df=0.5, wavelet="cmor1.5-1.0")

cmap = plt.get_cmap("tab10")
colors = [cmap(i) for i in range(num_states)]
grey = ["#E0E0E0"]
colors = grey + colors
state_labels = [f"S{i + 1}" for i in range(num_states)]

mosaic = [
    ["a1", "a1", "."],
    ["a2", "a2", "cbar1"],
    ["a3", "a3", "leg"],
    ["a4", "a4", "leg"],
    ["a5", "a5", "leg"],
    ["a6", "a6", "leg"], ]

fig, axes = plt.subplot_mosaic(
    mosaic,
    figsize=(FIG_WIDTH, 4),
    constrained_layout=True,
    gridspec_kw={
        "height_ratios": [0.6, 0.5, 0.2, 0.2, 0.2, 0.2, ],
        "width_ratios": [1, 1, 0.04, ],
    }
)

fig.set_constrained_layout_pads(
    w_pad=0.0,
    h_pad=0.0,
    wspace=0.01,
    hspace=0.001
)

##########
# simulated signal example
##########

axes["a1"].plot(trimmed_time_vec[:plot_t * fs], trimmed_signal[:plot_t * fs], lw=0.2, color="k")
axes["a1"].grid(False)
axes["a1"].set_frame_on(False)

axes["a1"].plot(trimmed_time_vec[:plot_t * fs],
                -1.7 + 1.2 * (trimmed_bursts[:plot_t * fs] * (trimmed_states[:plot_t * fs] == 1)), lw=0.2,
                color=colors[1])
axes["a1"].plot(trimmed_time_vec[:plot_t * fs],
                -2.5 + 1.2 * (trimmed_bursts[:plot_t * fs] * (trimmed_states[:plot_t * fs] == 2)), lw=0.2,
                color=colors[2])

axes["a1"].set_frame_on(False)
axes["a1"].tick_params(axis='both', which='both',
                       bottom=False, top=False, left=False, right=False,
                       labelbottom=False, labelleft=False)

############
# wavelet
############
im = axes["a2"].imshow(
    amp, extent=[t[0], t[-1], f[0], f[-1]],
    origin="lower", aspect="auto", cmap="viridis",
    vmin=amp.min(), vmax=amp.max()
)

axes["a2"].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)
axes["a2"].yaxis.set_major_locator(MultipleLocator(20))
axes["a2"].set_ylim(1, 55)
axes["a2"].set_ylabel("Freq.(Hz)")

# wavelet colorbar
cb = fig.colorbar(im, cax=axes["cbar1"])
cb.set_label("Amp.(a.u.)", rotation=270, labelpad=12)
cb.ax.tick_params(which="both", length=2, width=0.4, pad=1)

#############
# State time courses
############
# 1. true states

axes["a3"].stackplot(trimmed_time_vec[:plot_t * fs], states_onehot[:plot_t * fs].T, linewidth=0, step="mid", edgecolor="none", colors=colors,
                     labels=state_labels)
axes["a3"].set_ylabel(f"True States", rotation=0, labelpad=30, va="center")
axes["a3"].tick_params(axis='y', which='both', left=False, labelleft=False)

# 2. dpgmm (without imputation)
axes["a4"].stackplot(trimmed_time_vec[:plot_t * fs], dpgmm_states_onehot[:plot_t * fs].T, linewidth=0, step="mid", edgecolor="none", colors=colors, )
axes["a4"].set_ylabel(f"DP-GMM", rotation=0, labelpad=30, va="center")

# 3. hmm (without imputation)
axes["a5"].stackplot(trimmed_time_vec[:plot_t * fs], hmm_states_onehot[:plot_t * fs].T, linewidth=0, step="mid", edgecolor="none", colors=colors, )
axes["a5"].set_ylabel(f"HMM", rotation=0, labelpad=30, va="center")

# 4. threshold (without imputation)
axes["a6"].stackplot(trimmed_time_vec[:plot_t * fs], thresh_states_onehot[:plot_t * fs].T, linewidth=0, step="mid", edgecolor="none", colors=colors, )
axes["a6"].set_ylabel(f"Thresholding", rotation=0, labelpad=30, va="center")

for key in ["a3", "a4", "a5"]:
    axes[key].tick_params(axis='x', which='both', bottom=False, labelbottom=False, length=0)
    axes[key].tick_params(axis='y', which='both', left=False, labelleft=False)
    axes[key].set_yticks([])
    axes[key].set_xticks([])
    axes[key].spines["bottom"].set_visible(True)
    axes[key].set_ylim(0, 1)

for key in ["a1", "a2", "a3", "a4", "a5", "a6"]:
    axes[key].sharex(axes["a6"])

axes["a6"].set_xlabel("Time (s)")
axes["a6"].xaxis.set_major_locator(MultipleLocator(1.0))
axes["a6"].tick_params(axis='y', which='both', left=False, labelleft=False)
axes["a6"].set_ylim(0, 1)
axes["a6"].set_xlim(0, plot_t)
axes["a6"].spines["bottom"].set_visible(True)  # keep x-axis frame

# legend for states

handles, labels = axes["a3"].get_legend_handles_labels()

axes["leg"].axis("off")
state_legend = axes["leg"].legend(
    handles, labels,
    loc="center left",
    bbox_to_anchor=(-0.55, 0.7),
    bbox_transform=axes["leg"].transAxes,
    frameon=False,
    handlelength=1,
    borderaxespad=0.0,
)

state_legend.set_in_layout(False)

fig.savefig(
    fig_dir / f"{sim_cond}_before_imputation.svg",
    format="svg",
    bbox_inches="tight",
    bbox_extra_artists=(state_legend,),
    pad_inches=0.02,
)

fig.savefig(
    fig_dir / f"{sim_cond}_before_imputation.pdf",
    bbox_inches="tight",
    bbox_extra_artists=(state_legend,),
    pad_inches=0.02,

)

plt.show()

#%%
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm



methods = [
    "DP-GMM",
    "DP-GMM (Edge-removed)",
    "DP-GMM (Imputed & Edge-removed)",
]
titles = ["Raw", "Edge-removed", "Imputed"]

data = df.loc[
    df["method"].isin(methods), ["method", "burst_f", "snr", metric, "num_active_states"]
].copy()
data["frequency"] = data["burst_f"].map(relabel_freqs)
frequency_order = [relabel_freqs(pair) for pair in sorted(data["burst_f"].unique())]
snr_order = np.sort(data["snr"].dropna().unique())

def median_heatmap(method, column):
    return (
        data.loc[data["method"] == method]
        .pivot_table(
            index="frequency",
            columns="snr",
            values=column,
            aggfunc="median",
            observed=True,
        )
        .reindex(index=frequency_order, columns=snr_order)
    )


metric_maps = [median_heatmap(m, metric) for m in methods]
state_maps = [
    median_heatmap(m, "num_active_states")
    for m in methods
]

# Shared metric scale.
metric_min = 0
metric_max = 1


# Shared state-count scale: 1 = lower bound, 3 = neutral.
state_norm = TwoSlopeNorm(
    vmin=1,
    vcenter=true_num_states,
    vmax=num_states_dpgmm_upper,
)


fig, axes = plt.subplots(
    2, 3,
    figsize=(FIG_WIDTH, 5),
    constrained_layout=True,
)

common = dict(
    annot=True,
    annot_kws={"size": 5.2},
    cbar=False,
    xticklabels=[f"{snr:g}" for snr in snr_order],
    yticklabels=frequency_order,
)


# Top row: median metric.
for col, matrix in enumerate(metric_maps):
    sns.heatmap(
        matrix,
        ax=axes[0, col],
        fmt=".2f",
        cmap="viridis",
        vmin=metric_min,
        vmax=metric_max,
        **common,
    )
# Bottom row: median number of remaining active states.
for col, matrix in enumerate(state_maps):
    sns.heatmap(
        matrix,
        ax=axes[1, col],
        fmt=".1f",
        cmap="RdBu_r",
        norm=state_norm,
        **common,
    )




# Titles and axis labels.
for row in range(2):
    for col in range(3):
        a = axes[row, col]
        if row == 0:
            a.set_title(titles[col], fontsize=MY_FONT_SIZE)
        a.set_xlabel("")
        a.set_ylabel("")
        a.tick_params(
            axis="x",
            labelrotation=0,
            labelsize=MY_FONT_SIZE,
        )
        a.tick_params(
            axis="y",
            labelrotation=0,
            labelsize=MY_FONT_SIZE,
            labelleft=(col == 0),
        )

fig.supylabel("Frequency pair (Hz)", fontsize=MY_FONT_SIZE)
fig.supxlabel("SNR (dB)", fontsize=MY_FONT_SIZE)


# Metric colorbar beside Imputed.
metric_cax = axes[0, 2].inset_axes([1.04, 0, 0.05, 1])
metric_cbar = fig.colorbar(
    axes[0, 2].collections[0],
    cax=metric_cax,
)
metric_cbar.set_label(
    f"Median {metric_label}",
    rotation=270,
    labelpad=14,
    fontsize=MY_FONT_SIZE,
)

# State colorbar beside Imputed.
state_cax = axes[1, 2].inset_axes([1.04, 0, 0.05, 1])
state_cax.set_in_layout(False)

state_cbar = fig.colorbar(
    axes[1, 2].collections[0],
    cax=state_cax,
)
state_cbar.set_label(
    "Median active states",
    rotation=270,
    labelpad=14,
    fontsize=MY_FONT_SIZE,
)
state_cbar.set_ticks([1, true_num_states, num_states_dpgmm_upper])

for cbar in (metric_cbar, state_cbar):
    cbar.ax.tick_params(labelsize=MY_FONT_SIZE)


# Save.

for fmt in ("svg", "pdf"):
    fig.savefig(
        fig_dir / f"{sim_cond}_{metric}_heatmaps.{fmt}",
        bbox_inches="tight",
        pad_inches=0.03,
    )

plt.show()

#%%


snr_db = 6

snr_id = np.where(snr_range == snr_db)[0][0]
plot_f = (20, 40)
freq_id = np.where((np.array(freqs_dist_range) == np.array(plot_f)).all(axis=1))[0][0]


example_results = df.query("burst_f == @plot_f and snr == @snr_db")

# raw dpgmm
df_raw = example_results.query("method == 'DP-GMM'").iloc[sample_id]

#common across variants

signal = signal_sample[sample_id, freq_id, snr_id]
bursts = bursts_sample[sample_id, freq_id, snr_id]
true_states = states_sample[sample_id, freq_id, snr_id].astype(int)

num_emb = df_raw["num_emb"]
trimmed_signal = trim_data(signal, num_emb, verbose=False)
trimmed_bursts = trim_data(bursts, num_emb, verbose=False)
trimmed_states = trim_data(true_states, num_emb, verbose=False)
num_states_true = np.unique(trimmed_states).size

trimmed_time_vec = np.arange(len(trimmed_states)) / fs

#raw dpgmm results
est_params_raw = df_raw["est_params"]
weights_raw = est_params_raw["weights"]

num_states_raw = len(weights_raw)

est_states_raw = df_raw["est_states"]
est_states_onehot_raw = np.eye(num_states_raw)[est_states_raw]

untrunc_weights_raw = est_params_raw["untrunc_weights"]
untrunc_covs_raw = est_params_raw["untrunc_covs"]

#dpgmm with edge removal

df_no_edge = example_results.query("method == 'DP-GMM (Edge-removed)'").iloc[sample_id]

num_states_no_edge = df_no_edge["num_states"]
est_states_no_edge = df_no_edge["est_states"]
est_states_onehot_no_edge = np.eye(num_states_no_edge)[est_states_no_edge]

#dpgmm with edge removal and imputation

df_no_edge_imputed = example_results.query("method == 'DP-GMM (Imputed & Edge-removed)'").iloc[sample_id]

num_states_no_edge_imputed = df_no_edge_imputed["num_states"]
est_states_no_edge_imputed = df_no_edge_imputed["est_states"]

est_states_onehot_no_edge_imputed = np.eye(num_states_no_edge_imputed)[est_states_no_edge_imputed]

# Separate the configured bound, retained parameters, and occupied state labels.
# After imputation, occupied labels give the number of states still decoded.
diagnostic_rows = pd.DataFrame([df_raw, df_no_edge, df_no_edge_imputed]).set_index("method")
diagnostic_report = diagnostic_rows[["num_states", *report_metrics]].rename(
    columns={"num_states": "upper_bound"}
).copy()
diagnostic_report["retained_components"] = diagnostic_rows["est_params"].map(
    lambda params: len(params["weights"])
)
diagnostic_report["occupied_states"] = diagnostic_rows["est_states"].map(
    lambda states: np.unique(states).size
)
print(
    f"\nDP-GMM example: sample_id={sample_id}, "
    f"frequencies={plot_f[0]} & {plot_f[1]} Hz, SNR={snr_db} dB, "
    f"true states={num_states_true} (full evaluated sequences):"
)
print(diagnostic_report.to_string(float_format="{:.4f}".format))

# Paper summary: keep this condition separate and summarize across samples.
diagnostic_summary_data = example_results.loc[
    example_results["method"].isin(diagnostic_report.index)
].copy()
diagnostic_summary_data["retained_components"] = diagnostic_summary_data["est_params"].map(
    lambda params: len(params["weights"])
)
diagnostic_summary_data["occupied_states"] = diagnostic_summary_data["est_states"].map(
    lambda states: np.unique(states).size
)
diagnostic_summary = (
    diagnostic_summary_data.melt(
        id_vars="method",
        value_vars=[metric, "retained_components", "occupied_states"],
        var_name="quantity",
    )
    .groupby(["method", "quantity"], sort=False)["value"]
    .agg(n="count", median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75))
    .reset_index()
)
print("\nAcross simulation samples at the same condition (n, median, Q25, Q75):")
print(diagnostic_summary.to_string(index=False, float_format="{:.4f}".format))

#%%
def compute_cv(covs):
    """Compute the coefficient of variation of each covariance diagonal."""
    cv_vals = []

    for cov in covs:
        d = np.diag(cov)
        mu = np.mean(d)
        sd = np.std(d)
        cv_vals.append(sd / (mu + 1e-12))

    return np.array(cv_vals)

cv_vals = compute_cv(untrunc_covs_raw)



from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plot_t = 20
n_plot = int(round(plot_t * fs))

n_true_states = int(np.nanmax(trimmed_states)) + 1
n_raw_states = len(untrunc_weights_raw)

n_total_states = max(n_true_states, n_raw_states, num_states_raw)

cmap = plt.get_cmap("tab10")

colors = ["#E0E0E0"] + [cmap(i % 10) for i in range(n_total_states - 1)]

# One-hot true states
trimmed_states_onehot = np.eye(n_total_states)[trimmed_states.astype(int)]

# Wavelet
t, f, amp = compute_wavelet(
    trimmed_signal[:n_plot],
    fs,
    f_min=1,
    f_max=80,
    df=0.5,
    wavelet="cmor1.5-1.0",
)

mosaic = [
    ["a1", "a1", "."],
    ["a2", "a2", "cbar1"],
    ["a3", "a3", "leg"],
    ["a4", "a4", "leg"],
    ["a5", "a5", "leg"],
    ["a6", "a6", "leg"],
    ["gap", "gap", "."],
    ["b", "c", "."],
    ["d", "c", "."],
]

fig, axes = plt.subplot_mosaic(
    mosaic,
    figsize=(FIG_WIDTH + 0.5, 5),
    constrained_layout=True,
    gridspec_kw={
        "height_ratios": [0.7, 0.6, 0.15, 0.15, 0.15, 0.15, 0.08, 1.5, 0.225],
        "width_ratios": [1, 1, 0.04],
    },
)

fig.set_constrained_layout_pads(
    w_pad=0.0,
    h_pad=0.0,
    wspace=0.01,
    hspace=0.001,
)

axes["gap"].axis("off")

#############
# simulated signal example
############

time_plot = trimmed_time_vec[:n_plot]

axes["a1"].plot(
    time_plot,
    1.5 * trimmed_signal[:n_plot],
    lw=0.2,
    color="k",
)

axes["a1"].grid(False)
axes["a1"].set_frame_on(False)

scale_burst = 1.5

for state_id in range(1, min(n_true_states, 4)):
    axes["a1"].plot(
        time_plot,
        -2 - state_id
        + scale_burst * (
                trimmed_bursts[:n_plot]
                * (trimmed_states[:n_plot] == state_id)
        ),
        lw=0.2,
        color=colors[state_id],
    )

axes["a1"].tick_params(
    axis="both",
    which="both",
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False,
)

#########
# wavelet
#########

im = axes["a2"].imshow(
    amp,
    extent=[t[0], t[-1], f[0], f[-1]],
    origin="lower",
    aspect="auto",
    cmap="viridis",
    vmin=np.nanmin(amp),
    vmax=np.nanmax(amp),
)

axes["a2"].tick_params(
    axis="x",
    which="both",
    bottom=False,
    top=False,
    labelbottom=False,
    labeltop=False,
)
axes["a2"].yaxis.set_major_locator(MultipleLocator(20))
axes["a2"].set_ylim(1, 55)
axes["a2"].set_ylabel("Freq. (Hz)")

cb = fig.colorbar(im, cax=axes["cbar1"])
cb.set_label("Amp. (a.u.)", rotation=270, labelpad=12)
cb.ax.tick_params(which="both", length=2, width=0.4, pad=1)

# Robust SVG/PDF colorbar rendering
if hasattr(cb, "solids") and cb.solids is not None:
    cb.solids.set_edgecolor("face")
    cb.solids.set_rasterized(True)

##########
# true and estimated states
#########
axes["a3"].stackplot(
    time_plot,
    trimmed_states_onehot[:n_plot].T,
    linewidth=0, step="mid", edgecolor="none",
    colors=colors[:n_total_states],
)
axes["a3"].set_ylabel("True states", rotation=0, labelpad=30, va="center")

axes["a4"].stackplot(
    time_plot,
    est_states_onehot_raw[:n_plot].T,
    linewidth=0, step="mid", edgecolor="none",
    colors=colors[:est_states_onehot_raw.shape[1]],
    labels=[f"S{i + 1}" for i in range(est_states_onehot_raw.shape[1])],
)
axes["a4"].set_ylabel("Raw", rotation=0, labelpad=30, va="center")

axes["a5"].stackplot(
    time_plot,
    est_states_onehot_no_edge[:n_plot].T,
    linewidth=0, step="mid", edgecolor="none",
    colors=colors[:est_states_onehot_no_edge.shape[1]],
)
axes["a5"].set_ylabel("Edge-removed", rotation=0, labelpad=35, va="center")

axes["a6"].stackplot(
    time_plot,
    est_states_onehot_no_edge_imputed[:n_plot].T,
    linewidth=0, step="mid", edgecolor="none",
    colors=colors[:est_states_onehot_no_edge_imputed.shape[1]],
)
axes["a6"].set_ylabel(
    "Imputed",
    rotation=0,
    labelpad=30,
    va="center",
)

for key in ["a1", "a2", "a3", "a4", "a5", "a6"]:
    axes[key].sharex(axes["a1"])

for key in ["a3", "a4", "a5"]:
    axes[key].tick_params(
        axis="x",
        which="both",
        bottom=True,
        labelbottom=False,
        length=0,
    )
    axes[key].tick_params(
        axis="y",
        which="both",
        left=False,
        labelleft=False,
    )
    axes[key].set_yticks([])
    axes[key].set_xticks([])
    axes[key].set_ylim(0, 1)
    axes[key].set_xlim(0, plot_t)
    axes[key].spines["bottom"].set_visible(True)

axes["a6"].set_xlabel("Time (s)")
axes["a6"].xaxis.set_major_locator(MultipleLocator(1.0))
axes["a6"].tick_params(axis="y", which="both", left=False, labelleft=False)
axes["a6"].set_yticks([])
axes["a6"].set_ylim(0, 1)
axes["a6"].set_xlim(0, plot_t)
axes["a6"].spines["bottom"].set_visible(True)

##########
# Legend
#########
handles, labels = axes["a4"].get_legend_handles_labels()

axes["leg"].axis("off")
state_legend = axes["leg"].legend(
    handles,
    labels,
    loc="center left",
    bbox_to_anchor=(-1.4, 0.55),
    bbox_transform=axes["leg"].transAxes,
    frameon=False,
    handlelength=1,
    handletextpad=0.5,
    labelspacing=0.2,
    borderaxespad=0.0,
    ncol=1,
    columnspacing=0.4,
)

state_legend.set_in_layout(False)

##########
# mixture weights
# #########

x_states = np.arange(n_raw_states)

axes["b"].bar(
    x_states,
    untrunc_weights_raw,
    width=0.7,
    color=colors[:n_raw_states],
    edgecolor="k",
    linewidth=0.6,
    zorder=2,
)

for k, w in enumerate(untrunc_weights_raw):
    axes["b"].text(
        k,
        w + 0.02,
        f"{w * 100.0:.1f}%",
        ha="center",
        va="bottom",
        color="k",
        fontsize=MY_FONT_SIZE - 3,
    )

axes["b"].set_ylabel("Mixture weights")
axes["b"].set_xticks(x_states, labels=np.arange(1, n_raw_states + 1))
axes["b"].set_ylim(0, 1)
axes["b"].set_xlim(-0.5, n_raw_states - 0.5)
axes["b"].tick_params(axis="y")

##########
# CV of covariance diagonals
#########
y_states = np.arange(n_raw_states)

# Full components are ordered by descending mixture weight.
# Raw DP-GMM retains the leading num_states_raw components before edge removal.
removed_before_edge = y_states >= num_states_raw
panel_c_colors = [
    "black" if removed else colors[k]
    for k, removed in enumerate(removed_before_edge)
]

axes["c"].barh(
    y_states,
    cv_vals,
    color=panel_c_colors,
    alpha=1.0,
)

for i, cv in enumerate(cv_vals):

    axes["c"].text(
        cv + 0.02,
        i,
        f"{cv:.2f}",
        va="center",
        ha="left",
        fontsize=MY_FONT_SIZE - 1,
        color="black",
    )

axes["c"].set_yticks(y_states, labels=np.arange(1, n_raw_states + 1))
axes["c"].set_xlabel("CV of covariance diagonal")
axes["c"].set_ylabel("State ID")

axes["c"].set_ylim(-0.5, n_raw_states - 0.5)

# ------------------------------------------------------------
# D: covariance thumbnails
# ------------------------------------------------------------

thumb_half_width = 0.42
thumb_y0 = -0.02
thumb_y1 = 0.8

axes["d"].set_xlim(-0.5, n_raw_states - 0.5)

for k in range(n_raw_states):
    im_dpgmm_cov = axes["d"].imshow(
        untrunc_covs_raw[k],
        extent=(
            k - thumb_half_width,
            k + thumb_half_width,
            thumb_y0,
            thumb_y1,
        ),
        cmap="viridis",
        origin="upper",
        aspect="auto",
    )
    im_dpgmm_cov.set_rasterized(True)

axes["d"].set_xlabel("State ID")
axes["d"].set_xlim(-0.5, n_raw_states - 0.5)
axes["d"].set_ylim(thumb_y0, thumb_y1)
axes["d"].set_xticks([])
axes["d"].set_yticks([])

for spine in axes["d"].spines.values():
    spine.set_visible(False)

fig.align_xlabels([axes["c"], axes["d"]])

fig.canvas.draw()
fig.set_constrained_layout(False)

state_keys = ["a3", "a4", "a5", "a6"]

pos = [axes[k].get_position() for k in state_keys]

x0 = min(p.x0 for p in pos)
x1 = max(p.x1 for p in pos)
y_top = max(p.y1 for p in pos)
y_bottom = min(p.y0 for p in pos)

n_rows = len(state_keys)
gap = 0.0
total_h = y_top - y_bottom
row_h = (total_h - gap * (n_rows - 1)) / n_rows

y1 = y_top

for key in state_keys:
    y0 = y1 - row_h
    axes[key].set_position([x0, y0, x1 - x0, row_h])
    y1 = y0 - gap

wavelet_cbar_left_shift = 0.014

pos_cbar1 = axes["cbar1"].get_position()
axes["cbar1"].set_position([
    pos_cbar1.x0 - wavelet_cbar_left_shift,
    pos_cbar1.y0,
    pos_cbar1.width,
    pos_cbar1.height,
])

###########
# panel label
###########

import matplotlib.patheffects as pe
from matplotlib.font_manager import FontProperties

# Finalize constrained_layout positions before reading axes positions
fig.canvas.draw()
fig.set_constrained_layout(False)

panel_font = FontProperties(
    family="Helvetica Neue",
    weight="heavy",
    size=14,
)

label_kw = dict(
    fontproperties=panel_font,
    color="k",
    ha="left",
    va="top",
    path_effects=[
        pe.withStroke(linewidth=1, foreground="0.30")
    ],
)

# Get axes positions in figure coordinates
pos_a = axes["a1"].get_position()
pos_b = axes["b"].get_position()
pos_c = axes["c"].get_position()

# A and B aligned to the same absolute figure x-position
x_left = 0.03

fig.text(
    x_left,
    min(pos_a.y1 + 0.035, 0.985),
    "A",
    transform=fig.transFigure,
    **label_kw,
)

fig.text(
    x_left,
    pos_b.y1 + 0.020,
    "B",
    transform=fig.transFigure,
    **label_kw,
)

fig.text(
    pos_c.x0 - 0.09,
    pos_c.y1 + 0.020,
    "C",
    transform=fig.transFigure,
    **label_kw,
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------


fig.savefig(
    fig_dir / f"{sim_cond}_results_with_cv.pdf",
    bbox_inches="tight",
    bbox_extra_artists=[state_legend],
)

fig.savefig(
    fig_dir / f"{sim_cond}_results_with_cv.svg",
    bbox_inches="tight",
    bbox_extra_artists=[state_legend],
)

plt.show()

# %%

num_states = 3
snr_db = 6
snr_id = np.where(snr_range == snr_db)[0][0]
plot_f = (20, 40)
freq_id = np.where((np.array(freqs_dist_range) == np.array(plot_f)).all(axis=1))[0][0]

sample_id = 1

res = df.query("burst_f == @plot_f and snr == @snr_db")
dpgmm_res = res.query("method == 'DP-GMM (Imputed)'").iloc[sample_id]
num_emb = dpgmm_res.num_emb

signal = signal_sample[sample_id, freq_id, snr_id]
trimmed_signal = trim_data(signal, num_emb)
trimmed_states = trim_data(states_sample[sample_id, freq_id, snr_id], num_emb).astype(int)
trimmed_bursts = trim_data(bursts_sample[sample_id, freq_id, snr_id], num_emb)
trimmed_time_vec = np.arange(len(trimmed_states)) / fs

from nds_toolbox.preprocessing.features import (compute_tde, trim_data)

tde_signal = compute_tde(signal, num_emb)
dim_vec = np.arange(-num_emb // 2 + 1, num_emb // 2 + 1).astype(int)


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import numpy as np



cmap = plt.get_cmap("tab10")
colors = ["#E0E0E0"] + [cmap(i) for i in range(num_states)]
state_labels = [f"S{i + 1}" for i in range(num_states)]



plot_t = 12
n_plot = int(plot_t * fs)

# Better sample-aligned time vector
trimmed_time_vec = np.arange(len(trimmed_states)) / fs

time_plot = trimmed_time_vec[:n_plot]
signal_plot = trimmed_signal[:n_plot]
states_plot = trimmed_states[:n_plot]
bursts_plot = trimmed_bursts[:n_plot]

# Keep the orientation that worked
tde_plot = tde_signal[:n_plot, ::-1].T

# Symmetric robust color limits
vmax = np.nanpercentile(np.abs(tde_plot), 99.5)
vmin = -vmax

# ------------------------------------------------------------
# Figure layout: two plot axes + one dedicated colorbar axis
# ------------------------------------------------------------
fig = plt.figure(figsize=(FIG_WIDTH, 3))

gs = GridSpec(
    2, 2,
    figure=fig,
    width_ratios=[1.0, 0.025],
    height_ratios=[1.0, 0.8],
    hspace=0.34,
    wspace=0.035,
)

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
cax = fig.add_subplot(gs[1, 1])

# Empty upper-right cell: this preserves equal widths of ax0 and ax1
blank_ax = fig.add_subplot(gs[0, 1])
blank_ax.axis("off")

# ------------------------------------------------------------
# A. Signal
# ------------------------------------------------------------
ax0.plot(
    time_plot,
    signal_plot,
    lw=0.25,
    color="k",
)

ax0.plot(
    time_plot,
    2 + bursts_plot * (states_plot == 1),
    lw=0.35,
    color=colors[1],
)

ax0.plot(
    time_plot,
    3 + bursts_plot * (states_plot == 2),
    lw=0.35,
    color=colors[2],
)

ax0.set_xlim(time_plot[0], time_plot[-1])

ax0.tick_params(
    axis="both",
    which="both",
    bottom=True,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False,
)

for spine in ["top", "right", "left"]:
    ax0.spines[spine].set_visible(False)

ax0.spines["bottom"].set_linewidth(0.8)

ax0.text(
    -0.07, 1.05,
    "A",
    transform=ax0.transAxes,
    fontsize=14,
    fontweight="bold",
    va="bottom",
    ha="left",
)

# ------------------------------------------------------------
# B. Time-delay embedding
# ------------------------------------------------------------
pcm = ax1.pcolormesh(
    time_plot,
    dim_vec,
    tde_plot,
    shading="auto",
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
    rasterized=True,
)

ax1.set_xlim(time_plot[0], time_plot[-1])
ax1.set_ylim(dim_vec[0], dim_vec[-1])

ax1.set_ylabel("Lags")
ax1.set_xlabel("Time (s)")
ax1.set_title("Time-delay embedding", pad=4)

ax1.xaxis.set_major_locator(MaxNLocator(nbins=6))
ax1.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

for spine in ["top", "right"]:
    ax1.spines[spine].set_visible(False)

ax1.spines["bottom"].set_linewidth(0.8)
ax1.spines["left"].set_linewidth(0.8)

ax1.text(
    -0.07, 1.05,
    "B",
    transform=ax1.transAxes,
    fontsize=14,
    fontweight="bold",
    va="bottom",
    ha="left",
)

# ------------------------------------------------------------
# Colorbar: attached to dedicated cax, not ax1
# ------------------------------------------------------------
cbar = fig.colorbar(pcm, cax=cax)

cbar.set_label("Amplitude (a.u.)", rotation=270, labelpad=11)
cbar.outline.set_linewidth(0.6)
cbar.ax.tick_params(width=0.6, length=2.5)

fig.align_ylabels([ax0, ax1])



for fmt in ("svg", "pdf"):
    fig.savefig(
        fig_dir / f"tde_example.{fmt}",
        bbox_inches="tight",
        pad_inches=0.03,
    )

plt.show()


