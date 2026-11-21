import os
import pickle
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator

from nds_toolbox.preprocessing.features import (
    choose_embedding_dim,
    compute_wavelet,
    trim_data,
)
from nds_toolbox.sim.bursts.simulator import simulate_bursty_signal
from nds_toolbox.utils.helper import compare_decoding_performance

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
    "figure.autolayout": False,  # Use constrained_layout below.

    # --- DPI ---
    "savefig.dpi": 900,
    "figure.dpi": 300,
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
        project_root = candidate
        break
else:
    raise FileNotFoundError(f"Could not find the TDE-DP-GMM project root from {start_dir}")


n_jobs = os.cpu_count() or 1
print("n_jobs", n_jobs)

sim_cond = "3states_overparameterized_hmm"


data_dir = project_root / "data" / "simulations"
data_file = data_dir / f"{sim_cond}_data.npz"

fig_dir = project_root / "figures" / sim_cond
fig_dir.mkdir(parents=True, exist_ok=True)


####
data_dir.mkdir(parents=True, exist_ok=True)


# Set seeds for reproducibility.
seed = 2026
rng = np.random.default_rng(seed)

# ---parameters for burst simulator---
n_seconds = 60 * 3  # Total duration in seconds.
fs = 250
time_vec = np.arange(int(fs * n_seconds)) / fs

snr_db = 2

freq = np.array([20, 40])  # Three true states, including the noise state.


burst_amp_sigma = 0.1
tukey_alpha = 0.25
state_transition = "return_to_baseline"


beta = 1  # Pink noise.


# For burst segments, specify duration as the number of cycles.
burst_cycles = [3, 10]

# For noise segments, specify duration in seconds.
noise_duration = [0.2, 2.]


## simulate a bursty signal
signal_dict = simulate_bursty_signal(
    time_vec, fs, freq, burst_cycles, noise_duration,
    state_transition=state_transition,
    burst_type="sine",
    use_filter=True,
    snr_db=snr_db,
    burst_amp_sigma=burst_amp_sigma,
    beta=beta,
    tukey_alpha=tukey_alpha,
    rng=rng,
)

signal = signal_dict["signal"]
states = signal_dict["states"]
bursts = signal_dict["bursts"]
noise = signal_dict["noise"]


### save the data

np.savez_compressed(
    data_file,
    signal_sample=signal,
    states_sample=states,
    bursts_sample=bursts,
    noise_sample=noise,
)

print(f"Data saved as {data_file}")


###


raw_signal = signal.copy()  # Preserve the original amplitude for panel A.
signal = (signal - np.mean(signal)) / np.std(signal)  # Standardize model input.

num_emb = choose_embedding_dim(
    np.mean(freq),
    fs,
    min_cycles=2.5,
    ensure_odd=True,
)

# Symmetric TDE trims (E - 1) / 2 samples at each end.
n_embedded_dpgmm = len(signal) - (num_emb - 1)
num_states = int(np.ceil(np.log(n_embedded_dpgmm)))
# this sets K above the expected occupied clusters
print(f"Number of fitted components (both models): {num_states}")



data_info = {
    "signal": signal,
    "true_states": states,
    "burst_f": freq,
    "fs": fs,
    "snr": snr_db,
}


model_info = {
    "seed": seed,
    "num_states": num_states,
    "num_emb": num_emb,
    "use_model_tqdm": True,
    "filter_freq": None,
    "imputing_spurious_states": False,
    "remove_edge_states": True,
    "compute_summary_stats": True,
    "metrics": ("mcc", "ari", "kappa"),
    "debug_mode": False,
    "n_jobs": n_jobs,
}


dpgmm_res = compare_decoding_performance(
    data_info=data_info,
    model_info={
        **model_info,
        "use_dpgmm": True,
        "use_hmm": False,
        "truncate_weights": True,
    },
    verbose=True,
)

hmm_res = compare_decoding_performance(
    data_info=data_info,
    model_info={
        **model_info,
        "use_dpgmm": False,
        "use_hmm": True,
    },
    verbose=True,
)

# Each call returns a list of records; concatenate for pd.DataFrame(res).
res = dpgmm_res + hmm_res
performance_dir = project_root / "data" / "performance"
performance_dir.mkdir(parents=True, exist_ok=True)


# Save to performance_dir
save_path = performance_dir / f"{sim_cond}_res.pkl"
with open(save_path, "wb") as f:
    pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Performance results saved as {save_path}")


# %%

with open(save_path, "rb") as f:
    res = pickle.load(f)

res = pd.DataFrame(res)

print("\nAdjusted Rand index (ARI), by model variant:")
print(res[["method", "ari"]].to_string(index=False, float_format="{:.4f}".format))

# Preserve the original nonimputed selections. Use the exact variant names
# printed above to select imputed or edge-removed results instead.
dpgmm_method = "DP-GMM"
hmm_method = "HMM"
hmm_res = res.loc[res["method"] == hmm_method]
dpgmm_res = res.loc[res["method"] == dpgmm_method]

hmm_row = hmm_res.iloc[0]
dpgmm_row = dpgmm_res.iloc[0]

num_emb = int(dpgmm_row["num_emb"])
if int(hmm_row["num_emb"]) != num_emb:
    raise ValueError("The shared time axis requires the same embedding dimension for both models.")

gmm_states = np.asarray(dpgmm_row["est_states"]).astype(int, copy=False)
gmm_state_labels = np.asarray(dpgmm_row["state_labels"], dtype=int)
gmm_ari = dpgmm_row["ari"]
gmm_summary_stats = dpgmm_row["summary_stats"]
gmm_f, gmm_psds = gmm_summary_stats["spectra"]
gmm_params = dpgmm_row["est_params"]
untrunc_covs = np.asarray(gmm_params["untrunc_covs"])
untrunc_weights = np.asarray(gmm_params["untrunc_weights"])


hmm_states = np.asarray(hmm_row["est_states"]).astype(int, copy=False)
hmm_state_labels = np.asarray(hmm_row["state_labels"], dtype=int)
hmm_ari = hmm_row["ari"]
hmm_summary_stats = hmm_row["summary_stats"]
hmm_f, hmm_psds = hmm_summary_stats["spectra"]
hmm_params = hmm_row["est_params"]
hmm_covs = np.asarray(hmm_params["covs"])
stationary_dist = np.asarray(hmm_params["stationary_dist"])

trimmed_signal = trim_data(raw_signal, num_emb)
trimmed_states = trim_data(states, num_emb).astype(int)
trimmed_bursts = trim_data(bursts, num_emb)


if not (len(gmm_states) == len(hmm_states) == len(trimmed_states)):
    raise ValueError("Decoded states must align with the trimmed ground truth.")

# Rebase the trimmed example to zero; compute_wavelet uses the same convention.
trimmed_time_vec = np.arange(len(trimmed_states)) / fs

# Decoded states index parameter arrays; state_labels maps them to true-state
# labels, including distinct labels for unmatched extra states.
n_plot_states = max(
    int(trimmed_states.max()),
    int(gmm_state_labels.max()),
    int(hmm_state_labels.max()),
    len(untrunc_weights) - 1,
) + 1
hmm_states_onehot = np.eye(n_plot_states)[hmm_state_labels[hmm_states]]
gmm_states_onehot = np.eye(n_plot_states)[gmm_state_labels[gmm_states]]
states_onehot = np.eye(n_plot_states)[trimmed_states]

cmap = plt.get_cmap("tab10" if n_plot_states <= 11 else "tab20")
colors = ["#E0E0E0"] + [cmap(i % cmap.N) for i in range(n_plot_states - 1)]
state_labels = [f"S{i + 1}" for i in range(n_plot_states)]

plot_t = 20

t, f, amp = compute_wavelet(trimmed_signal[:plot_t * fs], fs, f_min=1, f_max=80, df=0.5, wavelet="cmor1.5-1.0")

mosaic = [["a1", "a1", "."],
          ["a2", "a2", "cbar1"],
          ["a3", "a3", "leg"],
          ["a4", "a4", "leg"],
          ["a5", "a5", "leg"],
          ["gap", "gap", "."],
          ["b", "c", "."],
          ["d", "e", "."],
          ]

fig, axes = plt.subplot_mosaic(
    mosaic,
    figsize=(FIG_WIDTH, 5.5),  # a hair taller
    constrained_layout=True,
    gridspec_kw={"height_ratios": [0.7, 0.6, 0.23, 0.23, 0.23, 0.08, 1.5, 0.2],
                 "width_ratios": [1, 1, 0.04]}
)
fig.set_constrained_layout_pads(
    w_pad=0.0,
    h_pad=0.0,
    wspace=0.01,
    hspace=0.001
)

axes["gap"].axis("off")

##########
# simulated signal example
##########

axes["a1"].plot(trimmed_time_vec[:plot_t * fs], 0.35 +(2 * trimmed_signal[:plot_t * fs]), lw=0.2, color="k")
axes["a1"].grid(False)

scale_burst = 1.8

axes["a1"].plot(trimmed_time_vec[:plot_t * fs],
                -3 + scale_burst * (trimmed_bursts[:plot_t * fs] * (trimmed_states[:plot_t * fs] == 1)), lw=0.2,
                color=colors[1])
axes["a1"].plot(trimmed_time_vec[:plot_t * fs],
                -4 + scale_burst * (trimmed_bursts[:plot_t * fs] * (trimmed_states[:plot_t * fs] == 2)), lw=0.2,
                color=colors[2])

axes["a1"].tick_params(axis='both', which='both',
                       bottom=False, top=False, left=False, right=False,
                       labelbottom=False, labelleft=False)

axes["a1"].grid(False)
axes["a1"].set_frame_on(False)

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
# #State time courses
############
#1. true states

axes["a3"].stackplot(trimmed_time_vec[:plot_t * fs], states_onehot[:plot_t * fs].T, linewidth=0, step = "mid", edgecolor="none", colors=colors)
axes["a3"].set_ylabel("True States", rotation=0, labelpad=30, va="center")
axes["a3"].tick_params(axis='y', which='both', left=False, labelleft=False)

#2. Selected DP-GMM variant
axes["a4"].stackplot(trimmed_time_vec[:plot_t * fs], gmm_states_onehot[:plot_t * fs].T, linewidth=0, step = "mid", edgecolor="none", colors=colors, )
axes["a4"].set_ylabel(dpgmm_method, rotation=0, labelpad=30, va="center")

#3. Selected HMM variant
axes["a5"].stackplot(trimmed_time_vec[:plot_t * fs], hmm_states_onehot[:plot_t * fs].T, linewidth=0, step = "mid", edgecolor="none", colors=colors,
                     labels=state_labels)
axes["a5"].set_ylabel(hmm_method, rotation=0, labelpad=30, va="center")

for key in ["a3", "a4"]:
    axes[key].tick_params(axis='x', which='both', bottom=False, labelbottom=False, length=0)
    axes[key].tick_params(axis='y', which='both', left=False, labelleft=False)
    axes[key].set_yticks([])
    axes[key].set_xticks([])
    axes[key].spines["bottom"].set_visible(True)
    axes[key].set_ylim(0, 1)

for key in ["a1", "a2", "a3", "a4"]:
    axes[key].sharex(axes["a5"])

axes["a5"].set_xlabel("Time (s)")
axes["a5"].xaxis.set_major_locator(MultipleLocator(1.0))
axes["a5"].tick_params(axis='y', which='both', left=False, labelleft=False)
axes["a5"].set_ylim(0, 1)
axes["a5"].set_xlim(0, plot_t)
axes["a5"].spines["bottom"].set_visible(True)  # keep x-axis frame

# legend for states

handles, labels = axes["a5"].get_legend_handles_labels()

axes["leg"].axis("off")
state_legend = axes["leg"].legend(
    handles,
    labels,
    loc="center left",
    bbox_to_anchor=(-0.53, 0.45),
    bbox_transform=axes["leg"].transAxes,
    frameon=False,
    handlelength=1,
    handletextpad=0.5,
    labelspacing=0.2,
    borderaxespad=0.0,
    ncol=2,
    columnspacing=0.4,
)

state_legend.set_in_layout(False)

##########
# mixture weights
###########

# The base DP-GMM variant retains the leading weight-sorted components.
# Map their matched state labels back to the original component order.
component_order = np.asarray(dpgmm_row["order"], dtype=int)
if not np.array_equal(np.asarray(gmm_params["covs"]), untrunc_covs[component_order]):
    raise ValueError("Selected DP-GMM variant needs its retained-component mapping for bar colors.")
component_labels = np.full(len(untrunc_weights), -1, dtype=int)
component_labels[component_order] = gmm_state_labels
# Give truncated components the remaining palette colors.
unassigned = component_labels < 0
unused_labels = np.setdiff1d(np.arange(len(colors)), gmm_state_labels)
component_labels[unassigned] = unused_labels[:np.count_nonzero(unassigned)]

axes["b"].bar(
    np.arange(len(untrunc_weights)),
    untrunc_weights,
    width=0.7,
    color=[colors[label] for label in component_labels],
    edgecolor="k",
    linewidth=0.6,
    zorder=2,
)

for k, w in enumerate(untrunc_weights):
    axes["b"].text(k, w + 0.02, "%.1f%%" % (w * 100.0), horizontalalignment="center", color="k",
                   fontsize=MY_FONT_SIZE - 3)
axes["b"].set_ylabel("Mixture weights")
axes["b"].set_xticks(np.arange(len(untrunc_weights)), labels=component_labels + 1)
axes["b"].set_ylim(0, 1)

#inset: dpgmm state psd
b_inset = axes["b"].inset_axes([0.50, 0.52, 0.46, 0.42])

for i, (state_f, state_psd) in enumerate(zip(gmm_f, gmm_psds)):
    b_inset.plot(state_f, state_psd, color=colors[gmm_state_labels[i]], lw=0.7)

b_inset.set_xlim(0, 55)
b_inset.set_ylim(0, 0.25)
b_inset.set_xlabel("Frequency (Hz)")
b_inset.set_ylabel("PSD (a.u.)")

#######
# stationary distribution
########
for k, w in enumerate(stationary_dist):
    axes["c"].bar(k, w, width=0.7, color=colors[hmm_state_labels[k]], zorder=1, align="center", edgecolor="k")
    axes["c"].text(k, w + 0.02, "%.1f%%" % (w * 100.0), horizontalalignment="center", color="k",
                   fontsize=MY_FONT_SIZE - 3)
axes["c"].set_ylabel("Stationary distribution")
axes["c"].set_xticks(np.arange(len(stationary_dist)), labels=hmm_state_labels + 1)
axes["c"].set_ylim(0, 1)

c_inset = axes["c"].inset_axes([0.50, 0.52, 0.46, 0.42])

for i, (state_f, state_psd) in enumerate(zip(hmm_f, hmm_psds)):
    c_inset.plot(state_f, state_psd, color=colors[hmm_state_labels[i]], lw=0.7)

c_inset.set_xlim(0, 55)
c_inset.set_ylim(0, 0.25)
c_inset.set_xlabel("Frequency (Hz)")
c_inset.set_ylabel("PSD (a.u.)")

############
# covariances
############


thumb_half_width = 0.42
thumb_y0 = 0.08
thumb_y1 = 0.92

# Align each covariance row with its corresponding bars.
for key in ["b", "d"]:
    axes[key].set_xlim(-0.5, len(untrunc_weights) - 0.5)
for key in ["c", "e"]:
    axes[key].set_xlim(-0.5, len(stationary_dist) - 0.5)

for k, cov in enumerate(untrunc_covs):
    # DP-GMM covariance matrices in the same weight order as panel B.
    im_dpgmm_cov = axes["d"].imshow(
        cov,
        extent=(k - thumb_half_width, k + thumb_half_width, thumb_y0, thumb_y1),
        cmap="viridis",
        origin="upper"

    )
    im_dpgmm_cov.set_rasterized(True)

for k, cov in enumerate(hmm_covs):
    # HMM covariance matrices in the same matched order as panel C.
    im_hmm_cov = axes["e"].imshow(
        cov,
        extent=(k - thumb_half_width, k + thumb_half_width, thumb_y0, thumb_y1),
        cmap="viridis",
        origin="upper",

    )
    im_hmm_cov.set_rasterized(True)

axes["d"].set_xlabel("DP-GMM states")
axes["e"].set_xlabel("HMM states")

for key in ["d", "e"]:
    axes[key].set_xticks([])
    axes[key].set_yticks([])
    axes[key].spines[:].set_visible(False)

###########
# panel label
###########

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

fig.canvas.draw()
fig.savefig(
    fig_dir / f"{sim_cond}_results.svg",
    bbox_inches="tight",
    bbox_extra_artists=(state_legend,),
    pad_inches=0.02,
)

fig.savefig(
    fig_dir / f"{sim_cond}_results.pdf",
    bbox_inches="tight",
    bbox_extra_artists=(state_legend,),
    pad_inches=0.02,
)

plt.show()
