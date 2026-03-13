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
        #alpha = numpyro.sample("alpha", dist.HalfCauchy(0.5))
        alpha = numpyro.sample("alpha", dist.Gamma(1.0, 2.0)) # mean = 0.5


    else:
        alpha = numpyro.deterministic("alpha", jnp.array(alpha_prior))

    # stick-breaking
    with numpyro.plate("v_plates", num_states - 1):
        v = numpyro.sample("v", dist.Beta(1.0, alpha))

    eps = 1e-12
    weights = stick_breaking(v)
    weights = jnp.clip(weights, eps, 1.0) #avoiding underflow
    weights = weights / jnp.sum(weights)

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
                                       1))  # maybe gamma is better? (gamma(1, 10))

            chol_corr = numpyro.sample("chol_corr",
                                       dist.LKJCholesky(num_dim, concentration=1.0))

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
        mixture_dist = dist.MixtureSameFamily(dist.Categorical(logits=jnp.log(weights)),
                                              dist.MultivariateNormal(mean, scale_tril=L_cov))
        numpyro.sample("obs", mixture_dist, obs=batch_data)


from numpyro.infer import SVI, TraceEnum_ELBO, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoDelta, AutoDiagonalNormal, AutoNormal
from jax import jit


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
               batch_size=2 ** 9,
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
        print(f"Best ELBO: {loss_best:.6g}")
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


def truncate(means, covs, weights, mass_threshold=0.99,
             remove_edge_states=True, edge_states_cv=0.2,
             verbose=False, return_info=False):

    w = jnp.asarray(weights)
    K = w.shape[0]

    # ----------------------------
    # (1) 99% mass truncation first
    # ----------------------------
    sort_idx = jnp.argsort(w)[::-1]
    w_sorted = w[sort_idx]
    cdf = jnp.cumsum(w_sorted)

    reached = cdf >= mass_threshold
    k0 = jnp.where(jnp.any(reached), jnp.argmax(reached) + 1, K)

    active_sorted_idx = sort_idx[:k0]
    active_idx = jnp.sort(active_sorted_idx)  # indices into original arrays

    # ----------------------------
    # (2) Remove edge states (within active set)
    # ----------------------------
    final_idx = active_idx
    removed_edge = jnp.array([], dtype=jnp.int32)

    if remove_edge_states:
        trunc_covs0 = covs[active_idx]
        diag = jnp.diagonal(trunc_covs0, axis1=-2, axis2=-1)  # (k0, D)
        mu = jnp.mean(diag, axis=-1)
        sd = jnp.std(diag, axis=-1)
        cv_diags = sd / (mu + 1e-12)

        keep_mask = cv_diags <= edge_states_cv
        removed_edge = jnp.where(~keep_mask)[0]

        if removed_edge.size > 0:
            # fail-safe: don't drop everything
            if not bool(jnp.any(keep_mask)):
                keep_mask = jnp.zeros_like(keep_mask, dtype=bool).at[jnp.argmin(cv_diags)].set(True)
                removed_edge = jnp.where(~keep_mask)[0]

            final_idx = active_idx[keep_mask]

    # ----------------------------
    # Final truncated params (after edge removal)
    # ----------------------------
    trunc_means = means[final_idx]
    trunc_covs  = covs[final_idx]

    # weights renormalized over survivors
    mass_kept_final = jnp.sum(w[final_idx])                 # <-- AFTER edge removal
    trunc_weights = w[final_idx] / (mass_kept_final + 1e-12)

    # ----------------------------
    # Info (AFTER edge removal)
    # ----------------------------
    if return_info or verbose:
        k_final = int(final_idx.shape[0])                   # <-- AFTER edge removal
        kept_mass = float(mass_kept_final)
        dropped_mass = float(1.0 - kept_mass)

        info = (
            "[truncate] "
            f"mass_threshold={float(mass_threshold):.3f} | kept {k_final}/{int(K)} "
            f"| mass kept={kept_mass:.4f} | mass dropped={dropped_mass:.4f}"
        )
        if remove_edge_states:
            info += f" | removed_edge={int(removed_edge.size)} (cv>{edge_states_cv})"

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

    log_joint = jnp.log(jnp.array(weights)) + log_likelihoods
    log_normalizer = logsumexp(log_joint, axis=-1, keepdims=True)
    probs = jnp.exp(log_joint - log_normalizer)

    return probs


def get_states(x, means, covs, weights):
    log_likelihoods = mvn_log_likelihood(x, means, covs)
    log_joint = jnp.log(jnp.array(weights)) + log_likelihoods
    states = jnp.argmax(log_joint, axis=-1)
    return states


