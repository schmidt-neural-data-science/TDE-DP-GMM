



from scipy.optimize import linear_sum_assignment


def match_states(
    one_hot_states1,
    one_hot_states2,
    verbose=False,
    *,
    return_state_labels=False,
):
    """Match states2 to states1, preserving all states2 columns."""
    states1 = np.asarray(one_hot_states1)
    states2 = np.asarray(one_hot_states2)

    if states1.ndim != 2 or states2.ndim != 2:
        raise ValueError("Inputs must be two-dimensional.")
    if states1.shape[0] != states2.shape[0]:
        raise ValueError("Inputs must have the same number of samples.")
    if min(states1.shape[1], states2.shape[1]) == 0:
        raise ValueError("Inputs must contain at least one state.")

    n_states1 = states1.shape[1]
    n_states2 = states2.shape[1]

    with np.errstate(divide="ignore", invalid="ignore"):
        full_corr = np.corrcoef(states1, states2, rowvar=False)

    corr_matrix = np.nan_to_num(
        full_corr[:n_states1, n_states1:],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    rows, cols = linear_sum_assignment(corr_matrix, maximize=True)
    unmatched = np.setdiff1d(np.arange(n_states2), cols)
    order = np.concatenate([cols, unmatched])

    # Parameter/column index -> matched ground-truth label.
    # Extra estimated states receive distinct labels beyond the true labels.
    state_labels = np.concatenate([
        rows,
        np.arange(n_states1, n_states1 + len(unmatched)),
    ])

    reordered = states2[:, order]

    if verbose:
        for i, j in zip(rows, cols):
            print(
                f"State {i + 1} matches State {j + 1} "
                f"(correlation: {corr_matrix[i, j]:.2f})"
            )
        print("Unmatched states2 columns:", unmatched)

    if return_state_labels:
        return order, reordered, state_labels

    return order, reordered


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




import numpy as np
from sklearn.metrics import matthews_corrcoef

def _infer_K(method, true_states, est_states, est_params, fallback_K: int):
    Ks = [int(fallback_K)]

    if true_states is not None and len(true_states) > 0:
        Ks.append(int(np.max(true_states)) + 1)

    if est_states is not None and len(est_states) > 0:
        Ks.append(int(np.max(est_states)) + 1)

    # Parameter-implied K (important after truncation)
    if method == "DP-GMM" and est_params is not None and "weights" in est_params:
        Ks.append(int(len(est_params["weights"])))
    if method == "HMM" and est_params is not None and "means" in est_params:
        Ks.append(int(est_params["means"].shape[0]))

    return max(Ks)


def _order_for_params(order, K_params: int):
    """
    Convert an 'order' over a larger state space (e.g., K_true) into a valid
    permutation over parameter indices [0..K_params-1].
    - drops indices >= K_params
    - appends any missing indices to keep length K_params
    """
    order = np.asarray(order, dtype=int)
    order_in = order[order < K_params]                 # keep only valid param indices
    missing = np.setdiff1d(np.arange(K_params), order_in, assume_unique=False)
    return np.concatenate([order_in, missing])



from sklearn.metrics import (
    adjusted_rand_score,
    cohen_kappa_score,
    matthews_corrcoef,
)


def return_performance_info(
    *,
    method: str,
    signal,
    fs,
    true_states,
    est_states,
    min_samples,
    num_states: int,
    num_emb: int,
    est_params: dict,
    loss: float,
    tde_signal=None,
    imputing_spurious_states=None,
    truncate_weights=None,
    remove_edge_states=False,
    compute_summary_stats=False,
    metrics=("mcc",),
    verbose=False,
):
    """
    Reorder model parameters and compute selected metrics after matching.

    metrics:
        "mcc", "ari" / "adjusted_rand_index", "kappa", or a sequence.

    Returned est_states index the reordered parameter arrays.
    state_labels maps those indices to ground-truth labels.
    matched_est_states contains those ground-truth-aligned labels.
    """
    metric_functions = {
        "mcc": matthews_corrcoef,
        "ari": adjusted_rand_score,
        "kappa": cohen_kappa_score,
    }
    aliases = {
        "adjusted_rand_index": "ari",
        "adjusted_rand_score": "ari",
        "cohen_kappa": "kappa",
        "cohen_kappa_score": "kappa",
    }

    if isinstance(metrics, str):
        metrics = (metrics,)
    elif metrics is None:
        metrics = ()

    requested_metrics = []
    for metric in metrics:
        name = metric.lower()
        name = aliases.get(name, name)
        if name not in metric_functions:
            raise ValueError(
                f"Unknown metric: {metric!r}. "
                "Choose 'mcc', 'ari', or 'kappa'."
            )
        if name not in requested_metrics:
            requested_metrics.append(name)

    est_states = np.asarray(est_states, dtype=int).copy()
    est_params = dict(est_params)

    # 1) Establish a consistent parameter and decoded-state order.
    if method == "DP-GMM":
        weights = np.asarray(est_params["weights"])
        means = np.asarray(est_params["means"])
        covs = np.asarray(est_params["covs"])

        weight_order = np.argsort(-weights)
        weights = weights[weight_order]
        means = means[weight_order]
        covs = covs[weight_order]

        # Old parameter index -> new parameter index.
        est_states = np.argsort(weight_order)[est_states]

        est_params.update(
            weights=weights,
            means=means,
            covs=covs,
        )

        if truncate_weights:
            (
                trunc_means,
                trunc_covs,
                trunc_weights,
                trunc_info,
            ) = truncate(
                means,
                covs,
                weights,
                remove_edge_states=remove_edge_states,
                verbose=verbose,
                return_info=True,
            )

            est_states = np.asarray(
                get_states(
                    tde_signal,
                    trunc_means,
                    trunc_covs,
                    trunc_weights,
                ),
                dtype=int,
            )

            est_params.update(
                weights=trunc_weights,
                means=trunc_means,
                covs=trunc_covs,
                untrunc_means=means,
                untrunc_covs=covs,
                untrunc_weights=weights,
                trunc_info=trunc_info,
                num_active_states=len(trunc_weights),
            )

        n_est_states = len(est_params["weights"])

    elif method == "HMM":
        initial_probs = np.asarray(est_params["initial_probs"])
        transition_probs = np.asarray(est_params["transition_probs"])
        means = np.asarray(est_params["means"])
        covs = np.asarray(est_params["covs"])

        est_params.update(
            initial_probs=initial_probs,
            transition_probs=transition_probs,
            means=means,
            covs=covs,
            stationary_dist=compute_stationary_distribution(
                transition_probs
            ),
        )
        n_est_states = means.shape[0]

    elif method == "Thresholding":
        # One noise state plus one state for each frequency band.
        n_est_states = max(
            int(est_states.max()) + 1,
            np.asarray(est_params["detect_k"]).size + 1,
        )

    else:
        raise ValueError(f"Unknown method: {method!r}")

    # 2) Optional imputation.
    if imputing_spurious_states:
        est_states = np.asarray(
            imputing_mode(est_states, min_samples=min_samples),
            dtype=int,
        )

    # 3) Match every parameter state, including unoccupied states.
    state_labels = np.arange(n_est_states)

    if true_states is not None:
        true_states = np.asarray(true_states, dtype=int)

        # Include gaps in the original ground-truth label sequence.
        n_true_states = int(true_states.max()) + 1
        true_onehot = np.eye(
            n_true_states, dtype=np.float32
        )[true_states]
        est_onehot = np.eye(
            n_est_states, dtype=np.float32
        )[est_states]

        order, reordered_onehot, state_labels = match_states(
            true_onehot,
            est_onehot,
            verbose=verbose,
            return_state_labels=True,
        )
        est_states = reordered_onehot.argmax(axis=1)

    elif method == "HMM":
        order = np.argsort(-est_params["stationary_dist"])
        est_states = np.argsort(order)[est_states]

    else:
        order = np.arange(n_est_states)

    # 4) Apply the same permutation to every state-indexed parameter.
    if method == "DP-GMM":
        for key in ("weights", "means", "covs"):
            est_params[key] = np.asarray(est_params[key])[order]

    elif method == "HMM":
        for key in (
            "initial_probs",
            "means",
            "covs",
            "stationary_dist",
        ):
            est_params[key] = np.asarray(est_params[key])[order]

        est_params["transition_probs"] = (
            est_params["transition_probs"][np.ix_(order, order)]
        )

    # Threshold parameters index frequency bands, not the complete
    # state space including noise. Preserve their original band order.

    # 5) Compute every selected metric after matching.
    matched_est_states = state_labels[est_states]
    scores = {name: None for name in metric_functions}

    if true_states is not None:
        for name in requested_metrics:
            scores[name] = metric_functions[name](
                true_states,
                matched_est_states,
            )
            if verbose:
                print(f"{name}: {scores[name]}")

    # Summary rows follow the same compact order as model parameters.
    summary_stats = (
        summarize_states(
            signal,
            est_states,
            fs,
            num_states=n_est_states,
        )
        if compute_summary_stats
        else None
    )

    return {
        "num_states": num_states,
        "num_emb": num_emb,
        "est_states": est_states,
        "matched_est_states": matched_est_states,
        "state_labels": state_labels,
        "order": order,
        "est_params": est_params,
        "summary_stats": summary_stats,
        **scores,
        "loss": loss,
    }

import time
from nds_toolbox.analysis.burst_analysis import optimize_threshold_params, thresholding_bursts
from nds_toolbox.preprocessing.features import (compute_tde, trim_data)
from nds_toolbox.models.hmm_pyro import fit_HMM, compute_viterbi_path, compute_stationary_distribution
from nds_toolbox.models.dpgmm_numpyro import fit_DPGMM, truncate, get_states


def prepare_tde_data(signal, true_states, num_emb):
    """
    Prepare time-delay embedded data for TDE-based models.

    Parameters
    ----------
    signal : array-like, shape (T,)
        Raw signal.
    true_states : array-like or None, shape (T,)
        Ground-truth states.
    num_emb : int
        Number of time-delay embeddings.

    Returns
    -------
    tde_signal : array, shape (T_trimmed, num_emb)
        Time-delay embedded signal.
    trimmed_signal : array, shape (T_trimmed,)
        Lag-zero / center signal corresponding to the embedded samples.
    trimmed_states : array or None, shape (T_trimmed,)
        Ground-truth states trimmed to match the embedded signal.
    """
    tde_signal = compute_tde(signal, num_emb, verbose=0)
    trimmed_signal = tde_signal[:, num_emb // 2]

    if true_states is None:
        trimmed_states = None
    else:
        trimmed_states = trim_data(true_states, num_emb, verbose=False)

    return tde_signal, trimmed_signal, trimmed_states


def compare_decoding_performance(
    *,
    data_info: dict,
    model_info: dict,
    metrics=None,
    verbose: bool = True,
):
    """
    Fit DP-GMM / finite Bayesian GMM, HMM, and/or thresholding models.

    Select evaluation metrics with metrics=("mcc", "ari", "kappa").
    An explicit selection takes precedence over model_info["metrics"].
    If metrics is None, use model_info["metrics"] or default to ("mcc",).
    Each returned record contains the selected scores; scores remain None
    when ground-truth states are unavailable.
    """
    import time
    import numpy as np

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    signal = data_info.get("signal")
    true_states = data_info.get("true_states")
    fs = data_info.get("fs")

    # ------------------------------------------------------------------
    # Model settings
    # ------------------------------------------------------------------
    seed = model_info.get("seed", 0)

    use_dpgmm = model_info.get("use_dpgmm", False)
    use_hmm = model_info.get("use_hmm", False)
    use_thresholding = model_info.get("use_thresholding", False)

    if metrics is None:
        metrics = model_info.get("metrics", ("mcc",))

    weight_prior = model_info.get(
        "weight_prior",
        "dirichlet_process",
    )

    if use_dpgmm and weight_prior not in (
        "dirichlet_process",
        "dirichlet_distribution",
    ):
        raise ValueError(
            f"Unknown weight_prior: {weight_prior!r}. "
            "Expected 'dirichlet_process' or "
            "'dirichlet_distribution'."
        )

    mixture_name = (
        "GMM"
        if weight_prior == "dirichlet_distribution"
        else "DP-GMM"
    )

    num_emb_default = model_info.get("num_emb")
    num_emb_dpgmm = model_info.get(
        "num_emb_dpgmm", num_emb_default
    )
    num_emb_hmm = model_info.get(
        "num_emb_hmm", num_emb_default
    )

    num_states = model_info.get("num_states")
    truncate_weights = model_info.get("truncate_weights")
    remove_edge_states = model_info.get(
        "remove_edge_states", False
    )
    imputing_spurious_states = model_info.get(
        "imputing_spurious_states", False
    )
    compute_summary_stats = model_info.get(
        "compute_summary_stats", False
    )

    use_model_tqdm = model_info.get("use_model_tqdm", True)
    debug_mode = model_info.get("debug_mode", False)
    n_jobs = model_info.get("n_jobs", 1)

    filter_freq = model_info.get("filter_freq")
    min_samples = model_info.get("min_samples")

    records = []

    # All variants pass through this helper, so metrics is forwarded
    # consistently to every return_performance_info call.
    def add_record(
        *,
        method,
        label,
        evaluation_signal,
        evaluation_states,
        est_states,
        est_params,
        num_emb,
        loss,
        fit_time,
        pred_time,
        model_result=None,
        tde_signal=None,
        impute=False,
        remove_edges=False,
        truncate_weights=None,
    ):
        performance_info = return_performance_info(
            method=method,
            signal=evaluation_signal,
            fs=fs,
            true_states=evaluation_states,
            est_states=est_states,
            min_samples=min_samples,
            num_states=num_states,
            num_emb=num_emb,
            est_params=est_params,
            loss=loss,
            tde_signal=tde_signal,
            imputing_spurious_states=impute,
            truncate_weights=truncate_weights,
            remove_edge_states=remove_edges,
            compute_summary_stats=compute_summary_stats,
            metrics=metrics,
            verbose=verbose,
        )

        record = _record(
            data_info=data_info,
            model_result=(
                model_result if model_result is not None else {}
            ),
            time_info={
                "fit_t": fit_time,
                "pred_t": pred_time,
                "total_time": fit_time + pred_time,
            },
            performance_info=performance_info,
        )

        records.append({**record, "method": label})

    # ------------------------------------------------------------------
    # DP-GMM / finite Bayesian GMM
    # ------------------------------------------------------------------
    if use_dpgmm:
        if verbose:
            print(f"--- Running {mixture_name} ---")

        tde_signal, trimmed_signal, trimmed_states = (
            prepare_tde_data(
                signal=signal,
                true_states=true_states,
                num_emb=num_emb_dpgmm,
            )
        )

        dpgmm_kwargs = {
            "data": tde_signal,
            "num_states": num_states,
            "weight_prior": weight_prior,
            "learn_mean": model_info.get("learn_mean", False),
            "learn_alpha": model_info.get("learn_alpha", True),
            "fixed_alpha": model_info.get("fixed_alpha", 1.0),
            "alpha_prior_scale": model_info.get(
                "alpha_prior_scale", 1.0
            ),
            "mean_prior_scale": model_info.get(
                "mean_prior_scale", 1.0
            ),
            "sigma_prior_scale": model_info.get(
                "sigma_prior_scale", 1.0
            ),
            "lkj_concentration": model_info.get(
                "lkj_concentration", 1.0
            ),
            "use_epoch_tqdm": False,
            "use_model_tqdm": use_model_tqdm,
            "main_seed": seed,
            "n_jobs": n_jobs,
            "verbose": False,
        }

        if debug_mode:
            dpgmm_kwargs.update(
                num_epochs=1,
                num_models=1,
            )

        t0 = time.perf_counter()
        dpgmm_result = fit_DPGMM(**dpgmm_kwargs)
        fit_time = time.perf_counter() - t0

        fitted_params = dpgmm_result["est_params"]
        loss_best = dpgmm_result["loss_best"]

        est_params = {
            "alpha": fitted_params["alpha"],
            "weights": fitted_params["weights"],
            "means": fitted_params["means"],
            "covs": fitted_params["covs"],
        }

        t1 = time.perf_counter()
        est_states = get_states(
            tde_signal,
            est_params["means"],
            est_params["covs"],
            est_params["weights"],
        )
        pred_time = time.perf_counter() - t1

        variants = [
            (mixture_name, False, False),
        ]

        if remove_edge_states:
            variants.append(
                (f"{mixture_name} (Edge-removed)", False, True)
            )

        if imputing_spurious_states:
            variants.append(
                (f"{mixture_name} (Imputed)", True, False)
            )

            if remove_edge_states:
                variants.append(
                    (
                        f"{mixture_name} (Imputed & Edge-removed)",
                        True,
                        True,
                    )
                )

        for label, impute, remove_edges in variants:
            add_record(
                method="DP-GMM",
                label=label,
                evaluation_signal=trimmed_signal,
                evaluation_states=trimmed_states,
                est_states=est_states,
                est_params=est_params,
                num_emb=num_emb_dpgmm,
                loss=loss_best,
                fit_time=fit_time,
                pred_time=pred_time,
                model_result=dpgmm_result,
                tde_signal=tde_signal,
                impute=impute,
                remove_edges=remove_edges,
                truncate_weights=truncate_weights,
            )

    # ------------------------------------------------------------------
    # HMM
    # ------------------------------------------------------------------
    if use_hmm:
        if verbose:
            print("--- Running HMM ---")

        tde_signal, trimmed_signal, trimmed_states = (
            prepare_tde_data(
                signal=signal,
                true_states=true_states,
                num_emb=num_emb_hmm,
            )
        )

        hmm_kwargs = {
            "data": tde_signal,
            "num_states": num_states,
            "seed": seed,
            "n_jobs": n_jobs,
            "use_model_tqdm": use_model_tqdm,
            "verbose": False,
        }

        if debug_mode:
            hmm_kwargs.update(
                num_models=1,
                num_epochs=1,
            )

        t0 = time.perf_counter()
        hmm_result = fit_HMM(**hmm_kwargs)
        fit_time = time.perf_counter() - t0

        est_params = hmm_result["est_params"]
        loss_best = hmm_result["loss_best"]

        t1 = time.perf_counter()
        est_states = compute_viterbi_path(
            tde_signal,
            **est_params,
        )
        pred_time = time.perf_counter() - t1

        variants = [("HMM", False)]

        if imputing_spurious_states:
            variants.append(("HMM (Imputed)", True))

        for label, impute in variants:
            add_record(
                method="HMM",
                label=label,
                evaluation_signal=trimmed_signal,
                evaluation_states=trimmed_states,
                est_states=est_states,
                est_params=est_params,
                num_emb=num_emb_hmm,
                loss=loss_best,
                fit_time=fit_time,
                pred_time=pred_time,
                model_result=hmm_result,
                impute=impute,
            )

    # ------------------------------------------------------------------
    # Thresholding
    # ------------------------------------------------------------------
    if use_thresholding:
        if verbose:
            print("--- Running Thresholding ---")

        threshold_signal = np.asarray(signal)
        threshold_states = (
            None
            if true_states is None
            else np.asarray(true_states)
        )

        filter_freq = np.atleast_1d(np.asarray(filter_freq))

        enveloped_signal_list = []
        threshold_params_array = np.zeros(
            (filter_freq.size, 2)
        )
        best_corr_list = []

        t0 = time.perf_counter()

        for i, freq in enumerate(filter_freq):
            enveloped_signal, threshold_params, best_corr = (
                optimize_threshold_params(
                    threshold_signal,
                    filter_freq=freq,
                    fs=fs,
                )
            )

            enveloped_signal_list.append(enveloped_signal)
            threshold_params_array[i, :] = np.asarray(
                threshold_params
            )
            best_corr_list.append(best_corr)

        fit_time = time.perf_counter() - t0

        t1 = time.perf_counter()

        est_states_list = []

        for i in range(filter_freq.size):
            est_states_i = thresholding_bursts(
                enveloped_signal_list[i],
                detect_k=threshold_params_array[i, 0],
                width_k=threshold_params_array[i, 1],
            )
            est_states_list.append(est_states_i)

        if filter_freq.size == 1:
            est_states = est_states_list[0]

        else:
            burst_states = np.asarray(est_states_list)
            envelopes = np.asarray(enveloped_signal_list)

            burst_mask = (burst_states != 0).astype(float)
            burst_scores = envelopes * burst_mask

            noise_row = (
                (burst_states == 0)
                .all(axis=0)
                .astype(float)[np.newaxis, :]
            )

            scores = np.concatenate(
                (noise_row, burst_scores),
                axis=0,
            )
            est_states = np.argmax(scores, axis=0)

        pred_time = time.perf_counter() - t1

        est_params = {
            "enveloped_signal": np.asarray(
                enveloped_signal_list
            ),
            "detect_k": threshold_params_array[:, 0],
            "width_k": threshold_params_array[:, 1],
            "best_corr": np.asarray(best_corr_list),
            "filter_freq": filter_freq,
        }

        variants = [("Thresholding", False)]

        if imputing_spurious_states:
            variants.append(
                ("Thresholding (Imputed)", True)
            )

        for label, impute in variants:
            add_record(
                method="Thresholding",
                label=label,
                evaluation_signal=threshold_signal,
                evaluation_states=threshold_states,
                est_states=est_states,
                est_params=est_params,
                num_emb=1,
                loss=threshold_params_array,
                fit_time=fit_time,
                pred_time=pred_time,
                impute=impute,
            )

    return records
