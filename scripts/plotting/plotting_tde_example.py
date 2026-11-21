# %% Settings and imports
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed, parallel_config
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from sklearn.metrics import adjusted_rand_score
from tqdm.auto import tqdm


#%%


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
    "axes.linewidth": 0.8,
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
    "savefig.dpi": 900,
    "figure.dpi": 300,
})


#%%
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


from nds_toolbox.analysis.utils import imputing_mode
from nds_toolbox.preprocessing.features import trim_data

sim_cond = "3states"

fs = 250  # Sampling frequency in Hz.
n_seconds = 60 * 3  # Total duration in seconds.
time_vec = np.arange(int(fs * n_seconds)) / fs

# Condition 1: frequency distances (3 states)
freqs_dist_range = [[10, 40], [15, 40], [20, 40], [25, 40], [30, 40], [35, 40]]  #30, 25, 20, 15, 10, 5Hz diff

# Condition 2: Signal-to-noise ratio (in dB)
snr_range = np.arange(-10, 11, 2)  # varying SNRs from -10 to 10 dB


fig_dir = project_dir / "figures" / sim_cond
fig_dir.mkdir(parents=True, exist_ok=True)

#%% load the simulated signals
signal_dir = project_dir / "data" / "simulations"
sim_data = np.load(f'{signal_dir}/{sim_cond}_data.npz')

#load the simulated data
signal_sample = sim_data['signal_sample']
states_sample = sim_data['states_sample']
bursts_sample = sim_data['bursts_sample']



# %% Load all frequency pairs and SNRs

file_path = project_dir / "data" / "performance" / f"results_{sim_cond}.pkl"
df = pd.concat(map(pd.DataFrame, pd.read_pickle(file_path)), ignore_index=True)
df["burst_f"] = df["burst_f"].apply(tuple)


#%% use one condition for a TDE figure

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

#%%

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import numpy as np



cmap = plt.get_cmap("tab10")
colors = ["#E0E0E0"] + [cmap(i) for i in range(num_states)]
state_labels = [f"S{i + 1}" for i in range(num_states)]



plot_t = 12
n_plot = int(plot_t * fs)

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

# A. Signal
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

# B. Time-delay embedding
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

fig.savefig(fig_dir/"tde_example.pdf", bbox_inches="tight")
fig.savefig(fig_dir/"tde_example.svg", dpi=600, bbox_inches="tight")

plt.show()
