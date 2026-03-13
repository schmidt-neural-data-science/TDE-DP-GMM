import numpy as np
import os
from nds_toolbox.sim.bursts.simulator import simulate_bursty_signal

#%%

simulation_condition = "2states"
figure_dir  = f"../figures/{simulation_condition}"
data_dir = "../data/simulations"
data_file = f"../data/simulations/{simulation_condition}_data.npz"


(os.makedirs
 (figure_dir, exist_ok=True))
os.makedirs(data_dir, exist_ok = True)


#global settings (used across all the simulations)

# Set seeds for reproducibility.
seed = 2025
rng = np.random.default_rng(seed)

# Simulation parameters.
n_samples = 5

n_seconds = 60*3   # Total duration in seconds.

burst_amp_sigma = 0.1
beta = 1 #pink noise

snr = 2

# For burst segments, specify duration as the number of cycles.
burst_cycles = [3, 7]

# For noise segments, specify duration in seconds.
noise_duration = [0.5, 3.]




# Condition 1: Frequency range (Hz)
freq_range = [10, 20, 30, 40, 50]

# Condition 2: Sampling frequency [fs]

fs_range = [100, 250, 500]



###################################################################


states_sample = np.empty((n_samples, len(freq_range), len(fs_range)), dtype=object)
bursts_sample = np.empty_like(states_sample)
noise_sample  = np.empty_like(states_sample)
signal_sample = np.empty_like(states_sample)

for sample_id in range(n_samples):
    for cond1_id, freq in enumerate(freq_range):
        for cond2_id, fs in enumerate(fs_range):

            time_vec = np.linspace(0, n_seconds, int(fs * n_seconds), endpoint=False)

            signal_dict = simulate_bursty_signal(
                time_vec, fs, freq, burst_cycles, noise_duration,
                burst_type="sine", snr_db=snr, beta=beta,
                burst_amp_sigma=burst_amp_sigma, rng=rng
            )

            states_sample[sample_id, cond1_id, cond2_id]  = signal_dict["states"]
            bursts_sample[sample_id, cond1_id, cond2_id]  = signal_dict["bursts"]
            noise_sample [sample_id, cond1_id, cond2_id]  = signal_dict["noise"]
            signal_sample[sample_id, cond1_id, cond2_id]  = signal_dict["signal"]



print("Signal shape", signal_sample.shape, "[samples, cond1, cond2]: the data points are stored as object")


### save the data

data_file = f"{data_dir}/{simulation_condition}_data.npz"
np.savez_compressed(data_file,
                    signal_sample=signal_sample,
                    states_sample=states_sample,
                    bursts_sample=bursts_sample,
                    noise_sample=noise_sample)

print(f"Data saved as {data_file}")