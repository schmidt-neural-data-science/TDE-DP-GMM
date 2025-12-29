import numpy as np
from scipy.optimize import linear_sum_assignment



def match_states(one_hot_states1, one_hot_states2, verbose=False):
    """
    Matches states from two one-hot encoded matrices by computing the correlation between
    each pair of columns and finding the best pairing using the Hungarian algorithm.
    """
    num_states_local = one_hot_states1.shape[1]
    correlation_matrix = np.zeros((num_states_local, num_states_local))

    for i in range(num_states_local):
        for j in range(num_states_local):
            std1 = np.std(one_hot_states1[:, i])
            std2 = np.std(one_hot_states2[:, j])
            # If either column is constant, assign a default correlation value to avoid div0
            if std1 == 0 or std2 == 0:
                correlation = 0.0
            else:
                correlation = np.corrcoef(one_hot_states1[:, i], one_hot_states2[:, j])[0, 1]
            correlation_matrix[i, j] = correlation

    # Use the Hungarian algorithm (linear_sum_assignment) to maximize the total correlation.
    row_indices, col_indices = linear_sum_assignment(-correlation_matrix)

    if verbose:
        print("\nBest mode matching:")
        for mode1_idx, mode2_idx in zip(row_indices, col_indices):
            print(f"Mode {mode1_idx + 1} in mode1 matches with Mode {mode2_idx + 1} in mode2 "
                  f"(Correlation: {correlation_matrix[mode1_idx, mode2_idx]:.2f})")

        print("\nCorrelation Matrix:")
        print(correlation_matrix)

    # 'order' defines the optimal reordering of states in one_hot_states2
    order = col_indices
    reordered_one_hot_states2 = one_hot_states2[:, order]

    return order, reordered_one_hot_states2



from collections.abc import Mapping

def _record(*, strict: bool = True, **sections: Mapping):
    """
    Combine multiple dict 'sections' into one record.

    Parameters
    ----------
    strict : bool
        If True, raise on duplicate keys across sections.
        If False, later sections overwrite earlier ones.
    **sections : dict
        Any number of named dicts, e.g. data_info=..., est_params=..., metrics=...

    Returns
    -------
    dict
        {
          'by_section': {'data_info': {...}, 'est_params': {...}, ...},
          'flat': {<all merged keys>},
        }
    """
    merged = {}
    for name, chunk in sections.items():
        merged.update(chunk)

    return merged

from nds_toolbox.analysis.burst_analysis import summarize_states
from nds_toolbox.analysis.utils import imputing_mode



from sklearn.metrics import matthews_corrcoef


def return_performance_info(*, method: str, signal, fs, true_states, est_states, min_samples,
                            num_states: int, num_emb: int, est_params: dict, loss: float, tde_signal = None, imputing_spurious_states = None, truncate_weights = None, compute_summary_stats = False):
    """
    Computes performance metrics and state summary after reordering estimated states to best match ground truth.
    """

    if true_states is None:
        mcc = None
        order = None
    else:
        est_states_onehot = np.eye(num_states)[est_states]
        true_states_onehot = np.eye(num_states)[true_states]
        order, reordered_est_states_onehot = match_states(true_states_onehot, est_states_onehot, verbose=False)
        est_states = np.argmax(reordered_est_states_onehot, axis=1)

    # Reorder estimated parameters to match true state order
    if method == "DP-GMM":
        alpha = est_params["alpha"]
        weights = est_params["weights"]
        means = est_params["means"]
        covs = est_params["covs"]

        if order is None:
            order = np.argsort(-weights)  # from large to small
            est_states = np.argsort(order)[est_states]

        weights = weights[order]
        means = means[order]
        covs = covs[order]

        if truncate_weights:
            print("Truncating weights")
            trunc_means, trunc_covs, trunc_weights, trunc_info = truncate(means, covs, weights, verbose=True, return_info= True)
            #override the estimated states with truncated params
            est_states = get_states(tde_signal, trunc_means, trunc_covs, trunc_weights)
            est_params = {
                "alpha": alpha,
                "weights": trunc_weights,
                "means": trunc_means,
                "covs": trunc_covs,
                "untrunc_means": means,
                "untrunc_covs": covs,
                "untrunc_weights": weights,
                "trunc_info": trunc_info,
            }


        else:

            est_params = {
                "alpha": alpha,
                "weights": weights,
                "means": means,
                "covs": covs,
            }


    elif method == "HMM":
        initial_probs, transition_probs, means, covs = est_params.values()
        stationary_dist = compute_stationary_distribution(transition_probs)

        if order is None:
            order = np.argsort(-stationary_dist)  # from large to small
            est_states = np.argsort(order)[est_states]

        initial_probs = initial_probs[order]
        transition_probs = transition_probs[order][:, order]
        means = means[order]
        covs = covs[order]
        stationary_dist = stationary_dist[order]


        est_params = {
            "initial_probs": initial_probs,
            "transition_probs": transition_probs,
            "means": means,
            "covs": covs,
            "stationary_dist":stationary_dist}

    elif method == "Thresholding":
        envelped_signal, detect_k, width_k = est_params.values()
        est_params = {
            "envelped_signal": envelped_signal,
            "detect_k": detect_k,
            "width_k": width_k,
        }


    if imputing_spurious_states:


        est_states = imputing_mode(est_states, min_samples = min_samples)


    if true_states is not None:
        mcc = matthews_corrcoef(true_states, est_states)


    if compute_summary_stats:
        summary_stats = summarize_states(signal, est_states, fs, num_states=num_states)
    else:
        summary_stats = None

    return {
        "num_states": num_states,
        "num_emb": num_emb,
        "est_states": est_states,
        "order": order,
        "est_params": est_params,
        "summary_stats": summary_stats,
        "mcc": mcc,
        "loss": loss,
    }

import time
from nds_toolbox.analysis.burst_analysis import optimize_threshold_params, thresholding_bursts
from nds_toolbox.preprocessing.features import (compute_tde, trim_data)
from nds_toolbox.models.hmm_pyro import fit_HMM, compute_viterbi_path, compute_stationary_distribution
from nds_toolbox.models.dpgmm_numpyro import fit_DPGMM, truncate, get_states



def compare_decoding_performance(*,
                                 data_info: dict,
                                 model_info: dict,
                                 verbose: bool = True):
    """
    Fits both DP-GMM and HMM models on a time-delay embedded signal and compares their decoding performance.
    """

    # Unpack data
    signal = data_info["signal"]
    true_states = data_info["true_states"]
    fs = data_info["fs"]

    # Unpack model info
    seed = model_info["seed"]

    use_dpgmm = model_info["use_dpgmm"]
    use_hmm = model_info["use_hmm"]


    num_emb = model_info["num_emb"]
    num_states = model_info["num_states"]
    truncate_weights = model_info["truncate_weights"]

    use_model_tqdm = model_info["use_model_tqdm"]

    use_thresholding = model_info["use_thresholding"]
    filter_freq = model_info["filter_freq"]
    min_samples = model_info["min_samples"]
    imputing_spurious_states = model_info["imputing_spurious_states"]
    compute_summary_stats = model_info["compute_summary_stats"]

    debug_mode = model_info["debug_mode"]
    n_jobs = model_info["n_jobs"]

    if true_states is not None:
        if num_states < np.unique(true_states).size:
            raise ValueError("num_states must be >= number of unique true states.")

    # Time-delay embedding
    tde_signal = compute_tde(signal, num_emb, verbose=0)
    trimmed_signal = tde_signal[:, num_emb // 2]  # lag0 = original signal
    if true_states is not None:
        trimmed_states = trim_data(true_states, num_emb, verbose=False)

    elif true_states is None:
        trimmed_states = None

    records = []

    if use_dpgmm:
        # ---- DP-GMM ----
        if verbose:
            print("--- Running DP-GMM ---")

        t0 = time.perf_counter()

        if debug_mode:
            dpgmm_result = fit_DPGMM(data=tde_signal,
                      num_states=num_states,
                      num_epochs= 1,
                      num_models= 1,
                      use_epoch_tqdm=False,
                      use_model_tqdm=use_model_tqdm,
                      main_seed=seed,
                      n_jobs = n_jobs,
                      verbose=False)
        else:
            dpgmm_result = fit_DPGMM(data=tde_signal,
                                     num_states=num_states,
                                     use_epoch_tqdm=False,
                                     n_jobs = n_jobs,
                                     use_model_tqdm=use_model_tqdm,
                                     main_seed=seed,
                                     verbose=False)


        fit_time = time.perf_counter() - t0

        est_params, loss_best, model_id_best, key_best, losses_all = dpgmm_result.values()

        alpha = est_params["alpha"]
        means = est_params["means"]
        covs = est_params["covs"]
        weights = est_params["weights"]

        t1 = time.perf_counter()
        est_states = get_states(tde_signal, means, covs, weights)
        pred_time = time.perf_counter() - t1

        dpgmm_record = _record(
            data_info=data_info,
            model_result=dpgmm_result,
            time_info={
                "fit_t": fit_time,
                "pred_t": pred_time,
                "total_time": fit_time + pred_time,
            },
            performance_info=return_performance_info(
                method="DP-GMM",
                signal=trimmed_signal,
                tde_signal = tde_signal,
                fs=fs,
                true_states=trimmed_states,
                num_states=num_states,
                num_emb = num_emb,
                est_states=est_states,
                est_params={
                    "alpha": alpha,
                    "weights": weights,
                    "means": means,
                    "covs": covs,
                },
                loss=loss_best,
                imputing_spurious_states=False,
                min_samples=min_samples,
                truncate_weights= truncate_weights,
                compute_summary_stats = compute_summary_stats,
            ),
            )
        records.append({"method": "DP-GMM", **dpgmm_record})

        if imputing_spurious_states:
            dpgmm_record = _record(
                data_info=data_info,
                model_result=dpgmm_result,
                time_info={
                    "fit_t": fit_time,
                    "pred_t": pred_time,
                    "total_time": fit_time + pred_time,
                },
                performance_info=return_performance_info(
                    method="DP-GMM",
                    signal=trimmed_signal,
                    tde_signal=tde_signal,
                    fs=fs,
                    true_states=trimmed_states,
                    num_states=num_states,
                    num_emb = num_emb,
                    est_states=est_states,
                    est_params={
                        "alpha": alpha,
                        "weights": weights,
                        "means": means,
                        "covs": covs,
                    },
                    loss=loss_best,
                    imputing_spurious_states=imputing_spurious_states,
                    compute_summary_stats = compute_summary_stats,
                    min_samples=min_samples,
                    truncate_weights = truncate_weights),
                )
            records.append({"method": "DP-GMM (Imputed)", **dpgmm_record})






    if use_hmm:

        # ---- HMM ----
        if verbose:
            print(f"--- Running HMM ---")

        t0 = time.perf_counter()

        if debug_mode:
            hmm_result = fit_HMM(data=tde_signal,
                                 num_states=num_states,
                                 num_models= 1,
                                 num_epochs= 1,
                                 seed=seed,
                                 n_jobs = n_jobs,
                                 use_model_tqdm=use_model_tqdm,
                                 verbose=False)

        else:
            hmm_result = fit_HMM(data=tde_signal,
                                 num_states=num_states,
                                 seed = seed,
                                 n_jobs = n_jobs,
                                 use_model_tqdm=use_model_tqdm,
                                 verbose=False)
        fit_time = time.perf_counter() - t0

        est_params, best_model_id, loss_best, loss_all = hmm_result.values()

        t1 = time.perf_counter()
        est_states = compute_viterbi_path(tde_signal, **est_params)
        pred_time = time.perf_counter() - t1

        hmm_record = _record(
            data_info=data_info,
            model_result=hmm_result,
            time_info={
                "fit_t": fit_time,
                "pred_t": pred_time,
                "total_time": fit_time + pred_time,
            },
            performance_info=return_performance_info(
                method=f"HMM",
                signal=trimmed_signal,
                fs=fs,
                true_states=trimmed_states,
                num_states=num_states,
                num_emb = num_emb,
                est_states=est_states,
                est_params=est_params,
                loss=loss_best,
                imputing_spurious_states=False,
                compute_summary_stats = compute_summary_stats,
                min_samples=min_samples,
            ),
        )
        records.append({"method": f"HMM", **hmm_record})

        if imputing_spurious_states:
            hmm_record = _record(
                data_info=data_info,
                model_result=hmm_result,
                time_info={
                    "fit_t": fit_time,
                    "pred_t": pred_time,
                    "total_time": fit_time + pred_time,
                },
                performance_info=return_performance_info(
                    method=f"HMM",
                    signal=trimmed_signal,
                    fs=fs,
                    true_states=trimmed_states,
                    num_states=num_states,
                    num_emb=num_emb,
                    est_states=est_states,
                    est_params=est_params,
                    loss=loss_best,
                    imputing_spurious_states=imputing_spurious_states,
                    compute_summary_stats = compute_summary_stats,
                    min_samples=min_samples,
                ),
            )
            records.append({"method": f"HMM (Imputed)", **hmm_record})







    if use_thresholding:
        # Threshold method
        if verbose:
            print("--- Running Thresholding ---")

        filter_freq = np.array(filter_freq)
        enveloped_signal_list = []
        threshold_params_array = np.zeros((np.array(filter_freq).size, 2))
        best_corr_list = []



        t0 = time.perf_counter()
        if filter_freq.size == 1:
            enveloped_signal, threshold_params, best_corr = optimize_threshold_params(trimmed_signal, filter_freq = filter_freq, fs = fs)
            enveloped_signal_list.append(enveloped_signal)
            threshold_params_array[0,:] = np.array(threshold_params)
            best_corr_list.append(best_corr)
            fit_time = time.perf_counter() - t0

        else:
            for i in range(filter_freq.size):
                enveloped_signal, threshold_params, best_corr = optimize_threshold_params(trimmed_signal, filter_freq = filter_freq[i], fs = fs)
                enveloped_signal_list.append(enveloped_signal)
                threshold_params_array[i,:] = np.array(threshold_params)
                best_corr_list.append(best_corr)
            fit_time = time.perf_counter() - t0


        est_states_list = []
        t1 = time.perf_counter()
        for i in range(filter_freq.size):
            est_states = thresholding_bursts(enveloped_signal_list[i], detect_k = threshold_params_array[i,0], width_k = threshold_params_array[i, 1])
            est_states_list.append(est_states)

        pred_time = time.perf_counter() - t1

        if filter_freq.size == 1:
            est_states = est_states_list[0]

        else:
            burst_states = np.array(est_states_list)
            envelopes = np.array(enveloped_signal_list)

            # mask envelopes by detected bursts (0 where no burst)
            burst_mask = (burst_states != 0).astype(float)
            burst_scores = envelopes * burst_mask  # (n_freqs, T)

            # noise row: 1 where no band has a burst, else 0
            noise_row = (burst_states == 0).all(axis=0).astype(float)[np.newaxis, :]  # (1, T)

            # stack noise (row 0) + burst bands (rows 1..n_freqs)
            scores = np.concatenate((noise_row, burst_scores), axis=0)  # (1 + n_freqs, T)

            # final state: 0 = noise, 1..n_freqs = band index with highest envelope
            est_states = np.argmax(scores, axis=0)


        est_params = {"enveloped_signal": np.array(enveloped_signal_list),
                      "detect_k": threshold_params_array[:,0],
                      "width_k": threshold_params_array[:, 1]}


        thresholding_record = _record(
            data_info=data_info,
            time_info={
                "fit_t": fit_time,
                "pred_t": pred_time,
                "total_time": fit_time + pred_time,
            },
            performance_info=return_performance_info(
                method="Thresholding",
                signal=trimmed_signal,
                fs=fs,
                true_states=trimmed_states,
                num_states=num_states,
                num_emb = num_emb,
                est_states=est_states,
                est_params=est_params,
                loss=threshold_params_array,
                imputing_spurious_states=False,
                compute_summary_stats = compute_summary_stats,
                min_samples=min_samples,
            ),
        )
        records.append({"method": "Thresholding", **thresholding_record})

        if imputing_spurious_states:
            thresholding_record = _record(
                data_info=data_info,
                time_info={
                    "fit_t": fit_time,
                    "pred_t": pred_time,
                    "total_time": fit_time + pred_time,
                },
                performance_info=return_performance_info(
                    method="Thresholding",
                    signal=trimmed_signal,
                    fs=fs,
                    true_states=trimmed_states,
                    num_states=num_states,
                    num_emb=num_emb,
                    est_states=est_states,
                    est_params=est_params,
                    loss=threshold_params_array,
                    imputing_spurious_states = imputing_spurious_states,
                    compute_summary_stats = compute_summary_stats,
                    min_samples=min_samples,
                ),
            )
            records.append({"method": "Thresholding (Imputed)", **thresholding_record})



    return records