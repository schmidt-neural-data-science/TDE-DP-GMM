from nds_toolbox.sim.bursts.simulator import simulate_bursty_signal
from nds_toolbox.preprocessing.features import (compute_tde, trim_data)
from nds_toolbox.preprocessing.features import choose_embedding_dim
from nds_toolbox.utils.helper import compare_decoding_performance
import numpy as np
import pickle
import os

n_jobs = os.cpu_count()
print("n_jobs", n_jobs)

sim_cond = "multiple_aperiodic_states"
data_dir = "../data/simulations"
data_file = f"{data_dir}/{sim_cond}_data.npz"

os.makedirs(data_dir, exist_ok=True)

# Set seed for reproducibility.
seed = 2026
rng = np.random.default_rng(seed)

# Simulation parameters
fs = 250
segment_seconds = 30
n_repetitions = 3

time_vec = np.arange(int(fs * segment_seconds)) / fs

snr_db = 2
freq = 20

burst_amp_sigma = 0.1
tukey_alpha = 0.25
state_transition = "return_to_baseline"

betas = [1, 0]  # Pink noise and brown noise

burst_cycles = [3, 10]
noise_duration = [0.2, 2.0]

# Simulate and interleave segments
signal_segments = []
state_segments = []
burst_segments = []
noise_segments = []

for repetition_id in range(n_repetitions):

    for beta_id, beta in enumerate(betas):
        signal_dict = simulate_bursty_signal(
            time_vec,
            fs,
            freq,
            burst_cycles,
            noise_duration,
            state_transition=state_transition,
            burst_type="sine",
            use_filter=True,
            snr_db=snr_db,
            burst_amp_sigma=burst_amp_sigma,
            noise_addition_mode="global_snr",
            beta=beta,
            tukey_alpha=tukey_alpha,
            rng=rng,
        )

        signal_segments.append(signal_dict["signal"])
        burst_segments.append(signal_dict["bursts"])
        noise_segments.append(signal_dict["noise"])

        # Assign four ground-truth states:
        # 0: Pink noise
        # 1: Brown noise
        # 2: Beta bursts + pink noise
        # 3: Beta bursts + brown noise
        states = np.where(
            signal_dict["states"] == 0,
            beta_id,
            beta_id + 2,
        )

        state_segments.append(states)

# Concatenate segments
signal = np.concatenate(signal_segments)
states = np.concatenate(state_segments)
bursts = np.concatenate(burst_segments)
noise = np.concatenate(noise_segments)

time_combined = np.arange(len(signal)) / fs

print("Signal shape:", signal.shape)
print("States:", np.unique(states))
print("Duration:", len(signal) / fs, "seconds")

assert np.allclose(signal, bursts + noise)

# Save simulation
np.savez_compressed(
    data_file,
    signal=signal,
    states=states,
    bursts=bursts,
    noise=noise,
    fs=fs,
)

print(f"Data saved as {data_file}")

# %%

import matplotlib.pyplot as plt

fig, axes = plt.subplots(
    2, 1,
    figsize=(12, 4),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]},
)

axes[0].plot(
    time_combined,
    signal,
    color="black",
    linewidth=0.5,
)

axes[0].set_ylabel("Amplitude")

axes[1].plot(
    time_combined,
    states,
    ".",
    markersize=0.5,
)

axes[1].set_yticks(
    [0, 1, 2, 3],
    [
        "Pink noise",
        "Brown noise",
        "Beta + pink",
        "Beta + brown",
    ],
)

# Mark transitions between pink and brown segments.
for t in np.arange(segment_seconds, 180, segment_seconds):
    for ax in axes:
        ax.axvline(
            t,
            color="gray",
            linestyle="--",
            linewidth=0.5,
        )

axes[-1].set_xlabel("Time (s)")
axes[-1].set_xlim(0, 180)

plt.tight_layout()
plt.show()

# %%

###


signal = (signal - np.mean(signal)) / np.std(signal)  # standarize

# Different embeddings for each model
num_emb_dpgmm = choose_embedding_dim(
    np.mean(freq),
    fs,
    min_cycles=2.5,
    ensure_odd=True,
)

num_states = int(np.ceil(np.log(len(signal))))  # E[K] = a ln n
# this sets K above the expected occupied clusters
print(f"Number of states: {num_states}")

data_info = {
    "signal": signal,
    "true_states": states,
    "burst_f": freq,
    "fs": fs,
    "snr": snr_db,
    "betas": betas, }

model_info = {
    "seed": seed,
    "use_dpgmm": True,
    "use_hmm": False,
    "num_states": num_states,

    "num_emb": num_emb_dpgmm,  # this is just fall back
    # Method-specific embeddings
    "num_emb_dpgmm": num_emb_dpgmm,

    "use_model_tqdm": True,
    "use_thresholding": False,
    "filter_freq": None,
    "imputing_spurious_states": True,
    "remove_edge_states": True,
    "compute_summary_stats": True,
    "min_samples": np.round((2 / np.max(freq)) * fs).astype(int),
    "truncate_weights": True,
    "debug_mode": False,
    "n_jobs": n_jobs
}

res = compare_decoding_performance(data_info=data_info, model_info=model_info, verbose=True)

print(res)

performance_dir = "../data/performance"
os.makedirs(performance_dir, exist_ok=True)

# Save to performance_dir
save_path = os.path.join(performance_dir, f"{sim_cond}_res.pkl")
with open(save_path, "wb") as f:
    pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Performance results saved as {save_path}")
