from tqdm.auto import tqdm

import numpy as np

import os

import jax
from jax import random
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpyro
import numpyro.distributions as dist

import jax.scipy.stats as jss
from numpyro.infer.initialization import init_to_median, init_to_feasible


# see https://github.com/luiarthur/TuringBnpBenchmarks/blob/master/src/dp-gmm/notebooks//dp_sb_gmm_pyro.ipynb

@jax.jit
def stick_breaking(v):
    batch_ndims = len(v.shape) - 1
    cumprod_one_minus_v = jnp.exp(jnp.log1p(-v).cumsum(-1))
    one_v = jnp.pad(v, [[0, 0]] * batch_ndims + [[0, 1]], constant_values=1)
    c_one = jnp.pad(cumprod_one_minus_v, [[0, 0]] * batch_ndims + [[1, 0]],
                    constant_values=1)
    return one_v * c_one


def dpgmm_model(data, *, num_states, batch_size=None, alpha_prior=1.0, learn_alpha=True, learn_mean=True):
    num_data, num_dim = data.shape

    if learn_alpha:
        # hyper prior
        alpha = numpyro.sample("alpha", dist.HalfCauchy(1))

    else:
        alpha = numpyro.deterministic("alpha", jnp.array(alpha_prior))

    # stick-breaking
    with numpyro.plate("v_plates", num_states - 1):
        v = numpyro.sample("v", dist.Beta(1.0, alpha))

    weights = stick_breaking(v)

    numpyro.deterministic("weights", weights)

    # --------  component‑specific parameters  ----------------------------
    if learn_mean:
        # sample a separate mean for every component
        with numpyro.plate("components", num_states):
            mean = numpyro.sample(
                "mean",
                dist.MultivariateNormal(jnp.zeros(num_dim),
                                        jnp.eye(num_dim))
            )
            sigma = numpyro.sample("sigma",
                                   dist.HalfCauchy(jnp.ones(num_dim)).to_event(
                                       1))

            chol_corr = numpyro.sample("chol_corr",
                                       dist.LKJCholesky(num_dim, concentration=1))

            L_cov = chol_corr * sigma[..., None]

    else:
        # fixed zero mean for each component
        mean = numpyro.deterministic("mean", jnp.zeros((num_states, num_dim)))
        with numpyro.plate("components", num_states):
            sigma = numpyro.sample("sigma",
                                   dist.HalfCauchy(jnp.ones(num_dim)).to_event(1))

            chol_corr = numpyro.sample("chol_corr",
                                       dist.LKJCholesky(num_dim, concentration=1.0))
            L_cov = numpyro.deterministic("L_cov", chol_corr * sigma[..., None])

    if batch_size is None:
        batch_size = num_data  # use all the data point all the times

    # mixture assignment + likelihood
    with numpyro.plate("data", num_data, subsample_size=batch_size) as ind:
        batch_data = data[ind]
        mixture_dist = dist.MixtureSameFamily(dist.Categorical(probs=weights),
                                              dist.MultivariateNormal(mean, scale_tril=L_cov))
        numpyro.sample("obs", mixture_dist, obs=batch_data)


from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal


from joblib import Parallel, delayed

def _fit_DPGMM(x,
               num_states,
               *,
               rng_key=None,
               seed=0,
               learn_mean=False,
               learn_alpha=True,
               num_epochs=3000,
               num_particles=1,
               batch_size=2 ** 10,
               alpha_prior=1.0,
               learning_rate=5e-2,
               use_epoch_tqdm=False
               ):
    x = jnp.array(x)
    guide = AutoNormal(dpgmm_model, init_loc_fn=init_to_feasible)

    # For numerical stability, normalize the scale by the total number of data points.
    scale_factor = 1.0 / float(x.size)
    scaled_model = numpyro.handlers.scale(dpgmm_model, scale=scale_factor)
    scaled_guide = numpyro.handlers.scale(guide, scale=scale_factor)

    optimizer = numpyro.optim.ClippedAdam(step_size=learning_rate,
                                          b1=0.95,
                                          b2=0.999,
                                          clip_norm=10, )

    elbo = Trace_ELBO(num_particles=num_particles, vectorize_particles=True)

    svi = SVI(scaled_model, scaled_guide, optimizer, loss=elbo)

    if rng_key is None:
        rng_key = random.PRNGKey(int(seed))

    svi_result = svi.run(rng_key, num_epochs, x, num_states=num_states, batch_size=batch_size, alpha_prior=alpha_prior,
                         learn_mean=learn_mean, learn_alpha=learn_alpha, progress_bar=use_epoch_tqdm)

    est_params = guide.median(svi_result.params)


    #compute ELBO estimate with 100 monte carlo samples
    elbo_eval = Trace_ELBO(num_particles = 100, vectorize_particles = True)


    final_loss = elbo_eval.loss(
        rng_key,
        svi_result.params,
        scaled_model,
        scaled_guide,
        x,
        num_states=num_states,
        batch_size=batch_size,
        alpha_prior=alpha_prior,
        learn_mean=learn_mean,
        learn_alpha=learn_alpha,)

    return est_params, np.array(svi_result.losses), final_loss

from tqdm_joblib import tqdm_joblib
def fit_DPGMM(
        *,
        data,
        num_states,
        learn_mean=False,
        learn_alpha=True,
        num_models=10,
        num_epochs=3000,
        num_particles= 10,
        batch_size=2**10,
        use_epoch_tqdm=False,
        use_model_tqdm=True,
        alpha_prior=1.0,
        learning_rate=5e-2,
        verbose=False,
        n_jobs=1,  # how many processes (-1 for all cores)
        main_seed=0,  # master seed for all runs
):
    x = np.array(data)

    # --- generate one integer seed per model run ---
    # Use numpy's SeedSequence for independent child seeds (robust & portable)
    ss = np.random.SeedSequence(int(main_seed))
    child_seeds = [s.generate_state(1, dtype=np.uint32)[0] for s in ss.spawn(num_models)]

    iterator = range(num_models)
    if use_model_tqdm and n_jobs == 1:
        iterator = tqdm(iterator, desc=f"Fitting {num_models} DPGMM models", unit="model", leave=False)

    loss_best = np.inf
    est_params_best = None
    best_model_id = None
    best_seed = None
    loss_all = []

    if n_jobs == 1:
        for i in iterator:
            params_i, losses_i, final_loss_i = _fit_DPGMM(
                x,
                num_states,
                rng_key=None,
                seed=int(child_seeds[i]),
                learn_mean=learn_mean,
                learn_alpha=learn_alpha,
                num_epochs=num_epochs,
                num_particles=num_particles,
                batch_size=batch_size,
                alpha_prior=alpha_prior,
                learning_rate=learning_rate,
                use_epoch_tqdm=use_epoch_tqdm,
            )
            loss_all.append(losses_i)
            run_best = final_loss_i

            if run_best < loss_best:
                loss_best = run_best
                est_params_best = params_i
                best_model_id = i
                best_seed = child_seeds[i]

    else:
        # ---- parallel with joblib ----
        indices = list(range(num_models))

        if use_model_tqdm:
            with tqdm_joblib(total=len(indices), desc=f"Fitting {num_models} DPGMM models"):
                results = Parallel(n_jobs=n_jobs)(
                    delayed(_fit_DPGMM)(
                        x,
                        num_states,
                        learn_mean=learn_mean,
                        learn_alpha=learn_alpha,
                        num_epochs=num_epochs,
                        num_particles=num_particles,
                        batch_size=batch_size,
                        alpha_prior=alpha_prior,
                        learning_rate=learning_rate,
                        use_epoch_tqdm=use_epoch_tqdm,
                        seed=int(child_seeds[i]),
                    )
                    for i in indices
                )

        else:
            if verbose:
                print("------running dpgmm-----")
            results = Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(_fit_DPGMM)(
                    x,
                    num_states,
                    learn_mean=learn_mean,
                    learn_alpha=learn_alpha,
                    num_epochs=num_epochs,
                    num_particles=num_particles,
                    batch_size=batch_size,
                    alpha_prior=alpha_prior,
                    learning_rate=learning_rate,
                    use_epoch_tqdm=use_epoch_tqdm,
                    seed=int(child_seeds[i]),
                )
                for i in indices
            )

        # collect in order
        results = list(results)
        for i, (params_i, losses_i, final_loss_i) in zip(indices, results):
            loss_all.append(losses_i)
            run_best = final_loss_i

            if run_best < loss_best:
                loss_best = run_best
                est_params_best = params_i
                best_model_id = i
                best_seed = child_seeds[i]

    if verbose:
        print(f"Best negative ELBO: {loss_best:.6g}")
        if best_model_id is not None:
            print(f"Best model ID: {best_model_id}")

    params = extract_params(est_params_best, learn_mean=learn_mean, learn_alpha=learn_alpha)
    alpha, weights, means, covs = params["alpha"], params["weights"], params["means"], params["covs"]

    if alpha is None:
        alpha = float(alpha_prior)
    if means is None:
        means = np.zeros((num_states, x.shape[1]))

    return {
        "est_params": {
            "alpha": alpha,
            "weights": weights,
            "means": means,
            "covs": covs,
        },
        "loss_best": loss_best,
        "model_id_best": best_model_id,
        "seed_best": int(best_seed) if best_seed is not None else None,
        "loss_all": loss_all,
    }


def extract_params(est_params, learn_mean=True, learn_alpha=True):
    # alpha
    alpha = None
    if learn_alpha:
        alpha = est_params["alpha"]

    # stick-breaking weights
    v = est_params["v"]
    weights = stick_breaking(v)

    # covariances
    chol_corr = est_params["chol_corr"]
    sigma = est_params["sigma"]
    L_covs = [jnp.diag(sig) @ L for sig, L in zip(sigma, chol_corr)]
    covs = jnp.array([L @ L.T for L in L_covs])

    means = None
    if learn_mean:
        means = est_params["mean"]

    return {
        "alpha": alpha,
        "weights": weights,
        "means": means,
        "covs": covs,
    }


import jax.numpy as jnp


def truncate(
    means,
    covs,
    weights,
    mass_threshold=0.99,
    remove_edge_states=True,
    edge_states_cv=0.1,
    verbose=False,
    return_info=False,
):
    """
    Truncate mixture components in two stages:

    1. Keep the smallest set of components whose cumulative weight mass
       reaches `mass_threshold`.
    2. Optionally remove "edge states" among those retained components,
       where edge states are defined by a diagonal covariance CV above
       `edge_states_cv`.

    Parameters
    ----------
    means : array, shape (K, ...)
    covs : array, shape (K, D, D)
    weights : array, shape (K,)
    mass_threshold : float
        Cumulative mass threshold for initial truncation.
    remove_edge_states : bool
        Whether to remove edge states after mass truncation.
    edge_states_cv : float
        Threshold on coefficient of variation of covariance diagonals.
    verbose : bool
        Whether to print a readable summary.
    return_info : bool
        Whether to return a summary string in addition to truncated params.

    Returns
    -------
    trunc_means, trunc_covs, trunc_weights
        Truncated parameters after renormalization.
    info : str, optional
        Human-readable summary if `return_info=True`.
    """
    means = jnp.asarray(means)
    covs = jnp.asarray(covs)
    w = jnp.asarray(weights)

    K = w.shape[0]

    # ----------------------------
    # (0) Basic validation
    # ----------------------------
    if means.shape[0] != K or covs.shape[0] != K:
        raise ValueError("means, covs, and weights must agree on the first dimension K.")

    if not (0.0 < mass_threshold <= 1.0):
        raise ValueError("mass_threshold must be in (0, 1].")

    if edge_states_cv < 0:
        raise ValueError("edge_states_cv must be nonnegative.")

    # ----------------------------
    # (1) Mass truncation
    # ----------------------------
    sort_idx = jnp.argsort(w)[::-1]
    w_sorted = w[sort_idx]
    cdf = jnp.cumsum(w_sorted)

    reached = cdf >= mass_threshold
    k0 = int(jnp.where(jnp.any(reached), jnp.argmax(reached) + 1, K))

    active_sorted_idx = sort_idx[:k0]
    active_idx = jnp.sort(active_sorted_idx)  # original component indices

    # diagnostics for stage 1
    mass_kept_stage1 = float(jnp.sum(w[active_idx]))
    mass_dropped_stage1 = float(1.0 - mass_kept_stage1)

    # ----------------------------
    # (2) Optional edge-state removal
    # ----------------------------
    final_idx = active_idx
    removed_edge_idx = jnp.array([], dtype=active_idx.dtype)
    kept_cv = None
    removed_cv = None
    cv_diags = None

    if remove_edge_states and active_idx.shape[0] > 0:
        trunc_covs_stage1 = covs[active_idx]  # (k0, D, D)
        diag = jnp.diagonal(trunc_covs_stage1, axis1=-2, axis2=-1)  # (k0, D)

        mu = jnp.mean(diag, axis=-1)
        sd = jnp.std(diag, axis=-1)
        cv_diags = sd / (mu + 1e-12)

        keep_mask = cv_diags <= edge_states_cv
        removed_edge_idx = active_idx[~keep_mask]   # original component indices

        # keep at least one state
        if not bool(jnp.any(keep_mask)):
            best_idx_within_active = jnp.argmin(cv_diags)
            keep_mask = jnp.zeros_like(keep_mask, dtype=bool).at[best_idx_within_active].set(True)
            removed_edge_idx = active_idx[~keep_mask]

        final_idx = active_idx[keep_mask]
        kept_cv = cv_diags[keep_mask]
        removed_cv = cv_diags[~keep_mask]

    # ----------------------------
    # (3) Final truncated params
    # ----------------------------
    trunc_means = means[final_idx]
    trunc_covs = covs[final_idx]

    mass_kept_final = jnp.sum(w[final_idx])
    trunc_weights = w[final_idx] / (mass_kept_final + 1e-12)

    # ----------------------------
    # (4) output log
    # ----------------------------
    info = None
    if return_info or verbose:
        k_mass = int(active_idx.shape[0])
        k_final = int(final_idx.shape[0])
        n_edge_removed = int(removed_edge_idx.shape[0])

        kept_mass_final = float(mass_kept_final)
        dropped_mass_final = float(1.0 - kept_mass_final)

        lines = [
            "[truncate]",
            f"  total components              : {K}",
            f"  mass threshold                : {float(mass_threshold):.3f}",
            "",
            "  Stage 1: mass truncation",
            f"    kept components             : {k_mass}",
            f"    kept original indices       : {list(map(int, active_idx.tolist()))}",
            f"    retained weight mass        : {mass_kept_stage1:.4f}",
            f"    dropped weight mass         : {mass_dropped_stage1:.4f}",
        ]

        if remove_edge_states:
            lines += [
                "",
                "  Stage 2: edge-state removal",
                f"    CV threshold                : {float(edge_states_cv):.3f}",
                f"    removed components          : {n_edge_removed}",
                f"    removed original indices    : {list(map(int, removed_edge_idx.tolist()))}",
            ]

            if cv_diags is not None:
                lines.append(
                    f"    CVs in active set           : {[round(float(x), 4) for x in cv_diags.tolist()]}"
                )

        lines += [
            "",
            "  Final result",
            f"    final components kept       : {k_final}",
            f"    final original indices      : {list(map(int, final_idx.tolist()))}",
            f"    final retained weight mass  : {kept_mass_final:.4f}",
            f"    final dropped weight mass   : {dropped_mass_final:.4f}",
        ]

        info = "\n".join(lines)

        if verbose:
            print(info)

    if return_info:
        return trunc_means, trunc_covs, trunc_weights, info

    return trunc_means, trunc_covs, trunc_weights


def mvn_log_likelihood(x, means, covs):
    # Computes the logpdf of a multivariate normal for each observation given a set of covariance matrices.
    def _mvn_logpdf(x, mean, cov):
        return jss.multivariate_normal.logpdf(x, mean, cov)

    # Vectorize over states for a fixed observation.
    _log_pdf_over_state = jax.vmap(_mvn_logpdf, in_axes=(None, 0, 0))

    # Vectorize over time (observations).
    log_likelihood = jax.vmap(_log_pdf_over_state, in_axes=(0, None, None))
    return log_likelihood(x, means, covs)


def get_state_probs(x, means, covs, weights):
    log_likelihoods = mvn_log_likelihood(x, means, covs)
    log_weights = jnp.log(jnp.clip(weights, 1e-12, 1.0))

    log_joint =  log_weights + log_likelihoods
    log_normalizer = logsumexp(log_joint, axis=-1, keepdims=True)
    probs = jnp.exp(log_joint - log_normalizer)

    return np.asarray(probs)


def get_states(x, means, covs, weights):
    log_likelihoods = mvn_log_likelihood(x, means, covs)
    log_weights = jnp.log(jnp.clip(weights, 1e-12, 1.0))
    log_joint = log_weights + log_likelihoods
    states = jnp.argmax(log_joint, axis=-1)
    return np.asarray(states)


