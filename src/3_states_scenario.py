
import os
import numpy as np
import pickle
from nds_toolbox.preprocessing.features import choose_embedding_dim
from nds_toolbox.utils.helper import compare_decoding_performance


from joblib import Parallel, delayed
import itertools
#%%

total_cores = os.cpu_count() or 1
inner_n_jobs = total_cores


print("total_cores:", total_cores)
print("inner_n_jobs:", inner_n_jobs)


#%%
sim_cond = "3states"
signal_dir = "../data/simulations"
performance_dir = "../data/performance"

sim_data = np.load(f'{signal_dir}/{sim_cond}_data.npz', allow_pickle=True)
signal_sample = sim_data['signal_sample']
states_sample = sim_data['states_sample']
print("shape of signal_sample", signal_sample.shape)

seed_base = 2025
fs = 250
num_states = 3

# Condition 1: frequency distances (3 states)
freqs_dist_range = [[10, 40], [15, 40], [20, 40], [25, 40], [30, 40], [35, 40]]  # 30, 25, 20, 15, 10, 5Hz diff

snr_range = np.arange(-10, 11, 2)


#%%

def _run_one(sample_id, freq_id, snr_id):
    freq = freqs_dist_range[freq_id]
    freq_for_emb = float(np.mean(freq))
    num_emb = choose_embedding_dim(freq_for_emb, fs, min_cycles=2.5)

    snr = snr_range[snr_id]
    sig = signal_sample[sample_id, freq_id, snr_id]
    sig = (sig - np.mean(sig)) / np.std(sig)  # standardize

    st = states_sample[sample_id, freq_id, snr_id].astype(int)

    # use different seed for each condition
    seed = int(seed_base + (sample_id * 10000 + freq_id * 100 + snr_id) * 997)

    data_info = {
        "signal": sig,
        "true_states": st,
        "burst_f": freq,
        "fs": fs,
        "snr": snr,
    }

    model_info = {
        "seed": seed,
        "use_dpgmm": True,
        "use_hmm": True,
        "num_states": num_states,
        "num_emb": num_emb,
        "use_model_tqdm": False,
        "use_thresholding": True,
        "filter_freq": np.array(freq),
        "imputing_spurious_states": True,
        "compute_summary_stats": False,
        "min_samples": int(np.round((2 / np.max(freq)) * fs)),
        "truncate_weights": False,
        "debug_mode": False,
        "n_jobs": inner_n_jobs
    }

    return compare_decoding_performance(data_info=data_info, model_info=model_info, verbose=True)


# Build the parameter grid
param_iter = itertools.product(
    range(signal_sample.shape[0]),
    range(signal_sample.shape[1]),
    range(signal_sample.shape[2])
)


print("num sims: ", int(signal_sample.size / signal_sample.shape[-1]))


from tqdm.auto import tqdm
total_sims = signal_sample.shape[0] * signal_sample.shape[1] * signal_sample.shape[2]
bar = tqdm(param_iter, total=total_sims, desc="Running sims", unit="sim")

all_results = [_run_one(i, j, k) for i, j, k in bar]

# Save
os.makedirs(performance_dir, exist_ok=True)
save_path = os.path.join(performance_dir, f"explore_embeddings_{sim_cond}.pkl")
with open(save_path, "wb") as f:
    pickle.dump(all_results, f)
print(f"Performance results saved as {save_path}")

#%%

"""
def main():
    torch.set_num_threads(1)
    pyro.clear_param_store()
    jax_config.update("jax_platform_name", "cpu")  # be explicit on macOS


    total_cores = os.cpu_count() or 1
    inner_n_jobs = 8
    outer_workers = 1

    print(f"total_cores={total_cores}, outer_workers={outer_workers}, inner_n_jobs={inner_n_jobs}")

    dask.config.set({'distributed.worker.daemon': False})

    # Create/close cluster & client cleanly
    with LocalCluster(
        n_workers=outer_workers,
        threads_per_worker=1,   # 1 thread per worker -> pure multiprocessing
        processes=True,         # True = real multiprocessing
        memory_limit="2.3GB",
        local_directory=local_dir,
        dashboard_address= None,
    ) as cluster, Client(cluster) as client:


        # --- your logic below here ---
        sim_cond = "3states"
        signal_dir = "../data/simulations"
        performance_dir = "../data/performance"

        sim_data = np.load(f'{signal_dir}/{sim_cond}_data.npz', allow_pickle=True)
        signal_sample = sim_data['signal_sample']
        states_sample = sim_data['states_sample']
        print("shape of signal_sample", signal_sample.shape)

        seed_base = 2025
        fs = 250
        num_states = 3

        # Condition 1: frequency distances (3 states)
        freqs_dist_range = [[10, 40], [15, 40], [20, 40], [25, 40], [30, 40], [35, 40]]  # 30, 25, 20, 15, 10, 5Hz diff

        snr_range = np.arange(-10, 11, 2)

        tasks_sample = []
        for sample_id in range(signal_sample.shape[0]):
            for freq_id in range(signal_sample.shape[1]):
                freq = freqs_dist_range[freq_id]
                freq_for_emb = np.mean(freq)  # take the average freqencies to optimize the number of emb
                num_emb = choose_embedding_dim(freq_for_emb, fs, min_cycles=2)

                for snr_id in range(signal_sample.shape[2]):
                    snr = snr_range[snr_id]
                    signal = signal_sample[sample_id, freq_id, snr_id]
                    signal = (signal - np.mean(signal)) / np.std(signal)

                    states = states_sample[sample_id, freq_id, snr_id].astype(int)

                    seed = int(seed_base + (sample_id * 10000 + freq_id * 100 + snr_id) * 997)

                    data_info = {
                        "signal": signal,
                        "true_states": states,
                        "burst_f": freq,
                        "fs": fs,
                        "snr": snr,
                    }


                    model_info = {
                        "seed": seed,
                        "use_dpgmm": True,
                        "use_hmm": True,
                        "num_states": num_states,
                        "num_emb": num_emb,
                        "use_model_tqdm": True,
                        "use_thresholding": False,
                        "filter_freq": None,
                        "imputing_spurious_states": True,
                        "min_samples": np.round((2 / np.max(freq)) * fs).astype(int),
                        "truncate_weights": False,
                        "debug_mode": False,
                        "n_jobs": inner_n_jobs
                    }

                    task = dask.delayed(compare_decoding_performance)(
                        data_info=data_info, model_info=model_info, verbose=True
                    )
                    tasks_sample.append(task)

        # Compute results
        futures = dask.persist(*tasks_sample)
        all_results = dask.compute(*futures)

        # Save
        os.makedirs(performance_dir, exist_ok=True)
        save_path = os.path.join(performance_dir, f"explore_embeddings_{sim_cond}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(all_results, f)
        print(f"Performance results saved as {save_path}")


if __name__ == "__main__":
    # On macOS/Windows the default is 'spawn'. Be explicit; ignore if already set.
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()






from nds_toolbox.sim.bursts.simulator import simulate_bursty_signal
from nds_toolbox.preprocessing.features import (compute_tde,trim_data)
from nds_toolbox.preprocessing.features import choose_embedding_dim
from nds_toolbox.utils.helper import compare_decoding_performance
import numpy as np
import pickle
import os
import dask
from dask.distributed import Client, LocalCluster



total_cores = os.cpu_count() or 1
inner_n_jobs = 1  # e.g., give half the cores to each model
outer_workers = total_cores


print(f"total_cores={total_cores}, outer_workers={outer_workers}, inner_n_jobs={inner_n_jobs}")

dask.config.set({'distributed.worker.daemon': False})

cluster = LocalCluster(
    n_workers=outer_workers,
    threads_per_worker=1,   # <- important
    processes=True          # <- important
)
client = Client(cluster)
print(client)



#%%

sim_cond = "3states"
signal_dir = "../data/simulations"
performance_dir = "../data/performance"




sim_data = np.load(f'{signal_dir}/{sim_cond}_data.npz', allow_pickle=True)
signal_sample = sim_data['signal_sample']
states_sample = sim_data['states_sample']

print("shape of signal_sample", signal_sample.shape)

##%%
seed_base = 2025
fs = 250
snr = 2
num_states = 2


# Condition 1: frequency distances (3 states)
freqs_dist_range = [[10, 40], [15, 40], [20, 40], [25, 40], [30, 40],[35, 40]]  #30, 25, 20, 15, 10, 5Hz diff

# Condition 2: Signal-to-noise ratio (in dB)
snr_range = np.arange(-10, 11, 2)  # varying SNRs from -10 to 10 dB


#%%

tasks_sample = []
num_states = 3  # set to ground truth

for sample_id in range(signal_sample.shape[0]):

    for freq_id in range(signal_sample.shape[1]):

        freq = freqs_dist_range[freq_id]
        freq_for_emb = np.mean(freq)  # take the average freqencies to optimize the number of emb
        num_emb = choose_embedding_dim(freq_for_emb, fs, min_cycles=2)

        for snr_id in range(signal_sample.shape[2]):

            snr = snr_range[snr_id]
            signal = signal_sample[sample_id, freq_id, snr_id]
            signal = (signal - np.mean(signal)) / np.std(signal)  # standarize the signal

            states = states_sample[sample_id, freq_id, snr_id].astype(int)

            seed = int(seed_base + (sample_id * 10000 + freq_id * 100 + snr_id) * 997)

            data_info = {
                "signal": signal,
                "true_states": states,
                "burst_f": freq,
                "fs": fs,
                "snr": snr, }

            model_info = {
                "seed": seed,
                "use_dpgmm": True,
                "use_hmm": True,
                "contrain_states": False,
                "num_states": num_states,
                "num_emb": num_emb,
                "use_model_tqdm": False,
                "use_thresholding": False,
                "filter_freq": None,
                "use_constrain_min_cycles": None,
                "constrained_freq": None,
                "removal_method": "merge", #wont be used
                "truncate_weights": None,
                "debug_mode": False,
                "n_jobs": inner_n_jobs,
                "verbose": False,
            }

            task = dask.delayed(compare_decoding_performance)(data_info=data_info, model_info=model_info,
                                                             verbose=True)

            tasks_sample.append(task)

#%%

all_results = dask.compute(*tasks_sample)

#%%
save_path = os.path.join(performance_dir, f"results_{sim_cond}.pkl")
with open(save_path, "wb") as f:
"""