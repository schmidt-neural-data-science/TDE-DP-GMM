from pyro.optim import ClippedAdam

import pyro
import pyro.distributions as dist
import torch
from tqdm.auto import tqdm

import numpy as np

from pyro import poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.infer.autoguide.initialization import init_to_median


from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed


def hmm_model(data, num_states, learn_mean=True, sequence_length=500, batch_size=2 ** 4):
    """
    Defines an HMM with multivariate normal emissions in Pyro.

    Parameters
    ----------
    data : torch.Tensor
        Input data of shape (T, D)
    num_states : int
        Number of hidden states
    learn_mean : bool
        Whether to learn the emission means (else fixed at zero)
    """
    num_data, num_dim = data.shape

    initial_probs = pyro.sample("initial_probs", dist.Dirichlet(torch.ones(num_states)))
    transition_probs = pyro.sample("transition_probs", dist.Dirichlet(torch.ones(num_states, num_states)).to_event(1))

    with pyro.plate("hidden_state", num_states):
        if learn_mean:
            mean = pyro.sample("mean", dist.MultivariateNormal(torch.zeros(num_dim), torch.eye(num_dim)))
        else:
            mean = pyro.deterministic("mean", torch.zeros(num_dim))

        sigma = pyro.sample("sigma", dist.HalfCauchy(torch.ones(num_dim)).to_event(1))
        chol_corr = pyro.sample("chol_corr", dist.LKJCholesky(num_dim, concentration=1.0))
        L_cov = torch.matmul(torch.diag_embed(sigma), chol_corr)

    if (batch_size is not None) and (sequence_length is not None):

        batched = data.unfold(0, size=sequence_length, step=sequence_length).permute(0, 2,
                                                                                     1)  # [num_batch, num_dim, num_sequence]; permute to [batch, sequence, dim]
        num_batch = batched.shape[0]
        with pyro.plate("sequences", size=num_batch, subsample_size=batch_size) as ind:
            batched_data = batched[ind]

            obs_dist = dist.MultivariateNormal(loc=mean, scale_tril=L_cov)
            hmm_dist = dist.DiscreteHMM(initial_probs.log(), transition_probs.log(), obs_dist)

            pyro.sample("data", hmm_dist, obs=batched_data)

    else:
        obs_dist = dist.MultivariateNormal(loc=mean, scale_tril=L_cov)
        hmm_dist = dist.DiscreteHMM(initial_probs.log(), transition_probs.log(), obs_dist, duration=num_data)
        pyro.sample("data", hmm_dist, obs=data)


from pyro.util import set_rng_seed, get_rng_state, set_rng_state


def _fit_HMM(x,
             num_states,
             *,
             learn_mean=False,
             sequence_length=500,
             batch_size=2 **4,
             num_epochs=3000,
             num_particles=8,
             learning_rate=5e-2,
             verbose=False,
             use_epoch_tqdm=False,
             seed=0):
    """
    Single HMM fit for one seed.
    """
    pyro.clear_param_store()
    pyro.set_rng_seed(int(seed))

    x_tensor = x

    exposed_params = ["initial_probs", "transition_probs", "mean", "sigma", "chol_corr"]
    guide = AutoNormal(poutine.block(hmm_model, expose=exposed_params),
                       init_loc_fn=init_to_median)

    scale_factor = 1.0 / (x_tensor.shape[0] * x_tensor.shape[1])
    scaled_model = poutine.scale(hmm_model, scale=scale_factor)
    scaled_guide = poutine.scale(guide, scale=scale_factor)

    optimizer = pyro.optim.Adam({"lr": learning_rate})
    elbo = Trace_ELBO(num_particles=num_particles)
    svi = SVI(scaled_model, scaled_guide, optimizer, loss=elbo)

    iterator = range(1, num_epochs + 1)
    if use_epoch_tqdm:
        iterator = tqdm(iterator, desc="Training epochs", unit="epoch")

    losses = []
    for step in iterator:
        loss = svi.step(x_tensor, num_states, learn_mean, sequence_length, batch_size)
        losses.append(loss)
        if verbose and step % 100 == 0:
            msg = f"ELBO loss [{step:4d}]: {loss:.6f}"
            tqdm.write(msg) if use_epoch_tqdm else print(msg)

    param_state = pyro.get_param_store().get_state()

    return param_state, np.array(losses)


def fit_HMM(
        *,
        data,
        num_states,
        learn_mean=False,
        sequence_length=500,
        batch_size=2 ** 4,
        num_models=8,
        num_epochs=1000,
        num_particles=1,
        learning_rate=5e-2,
        verbose=False,
        use_epoch_tqdm=False,
        use_model_tqdm=True,
        seed=0,
        n_jobs=1,
):
    x_tensor = torch.tensor(data, dtype=torch.float32)
    set_rng_seed(int(seed))

    # --- generate one integer seed per model run ---
    ss = np.random.SeedSequence(int(seed))
    child_seeds = [s.generate_state(1, dtype=np.uint32)[0] for s in ss.spawn(num_models)]

    param_states_all = []
    loss_all = []

    if n_jobs == 1:
        iterator = range(num_models)
        if use_model_tqdm:
            iterator = tqdm(iterator, desc=f"Fitting {num_models} HMM models",
                            unit="model", leave=False)

        for i in iterator:
            param_state_i, losses_i = _fit_HMM(
                x_tensor,
                num_states,
                learn_mean=learn_mean,
                sequence_length=sequence_length,
                batch_size=batch_size,
                num_particles=num_particles,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                verbose=verbose,
                use_epoch_tqdm=use_epoch_tqdm,
                seed=int(child_seeds[i]),
            )
            param_states_all.append(param_state_i)
            loss_all.append(losses_i)

    else:
        indices = list(range(num_models))

        if use_model_tqdm:
            with tqdm_joblib(total=len(indices), desc=f"Fitting {num_models} HMM models"):
                results = Parallel(n_jobs=n_jobs)(
                    delayed(_fit_HMM)(
                        x_tensor,
                        num_states,
                        learn_mean=learn_mean,
                        sequence_length=sequence_length,
                        batch_size=batch_size,
                        num_particles=num_particles,
                        num_epochs=num_epochs,
                        learning_rate=learning_rate,
                        verbose=verbose,
                        use_epoch_tqdm=use_epoch_tqdm,
                        seed=int(child_seeds[i]),
                    )
                    for i in indices
                )
        else:
            if verbose:
                print("---------running HMM -----------")
            results = Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(_fit_HMM)(
                    x_tensor,
                    num_states,
                    learn_mean=learn_mean,
                    sequence_length=sequence_length,
                    batch_size=batch_size,
                    num_particles=num_particles,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    verbose=verbose,
                    use_epoch_tqdm=use_epoch_tqdm,
                    seed=int(child_seeds[i]),
                )
                for i in indices
            )

        results = list(results)
        for param_state_i, losses_i in results:
            param_states_all.append(param_state_i)
            loss_all.append(losses_i)

    exposed_params = ["initial_probs", "transition_probs", "mean", "sigma", "chol_corr"]
    guide_eval = AutoNormal(poutine.block(hmm_model, expose=exposed_params),
                            init_loc_fn=init_to_median)

    scale_factor = 1.0 / (x_tensor.shape[0] * x_tensor.shape[1])
    scaled_model_eval = poutine.scale(hmm_model, scale=scale_factor)
    scaled_guide_eval = poutine.scale(guide_eval, scale=scale_factor)

    optimizer_eval = pyro.optim.Adam({"lr": learning_rate})
    elbo_eval = Trace_ELBO(num_particles=100)  # use 100 MC particle for evaluation
    svi_eval = SVI(scaled_model_eval, scaled_guide_eval, optimizer_eval, loss=elbo_eval)
    base_rng_state = get_rng_state()

    # Sequentially compute an ELBO estimate for each model---
    loss_best = np.inf
    best_model_id = None
    best_param_state = None

    for i, param_state_i in enumerate(param_states_all):
        pyro.clear_param_store()
        set_rng_state(base_rng_state)
        pyro.get_param_store().set_state(param_state_i)

        elbo_i = float(
            svi_eval.evaluate_loss(
                x_tensor,
                num_states,
                learn_mean,
                sequence_length,  # sequence_length
                batch_size,  # batch_size
            )
        )

        if elbo_i < loss_best:
            loss_best = elbo_i
            best_model_id = i
            best_param_state = param_state_i

    if verbose:
        print(f"Best negative ELBO: {loss_best:.6g}")
        print(f"Best model ID: {best_model_id}")

    # Extract median of posterior for the best run ---
    pyro.clear_param_store()
    pyro.get_param_store().set_state(best_param_state)
    # sample latent sites once under the best parameters
    est_latents = guide_eval(x_tensor, num_states, learn_mean, None, None)

    if learn_mean:
        initial_probs, transition_probs, means, covs = extract_params(est_latents,
                                                                      learn_mean=learn_mean)
    else:
        initial_probs, transition_probs, covs = extract_params(est_latents,
                                                               learn_mean=learn_mean)
        means = np.zeros((num_states, covs.shape[1]))

    return {
        "est_params": {
            "initial_probs": initial_probs,
            "transition_probs": transition_probs,
            "means": means,
            "covs": covs,
        },
        "best_model_id": best_model_id,
        "loss_best": loss_best,
        "loss_all": loss_all,
    }


def extract_params(est_params, learn_mean=True):
    """
    Extract latent parameters from trained guide.

    Returns
    -------
    initial_probs : np.ndarray
    transition_probs : np.ndarray
    means : np.ndarray
    covs : np.ndarray
    """

    initial_probs = est_params["initial_probs"].data.cpu().numpy()
    transition_probs = est_params["transition_probs"].data.cpu().numpy()
    sigma = est_params["sigma"].data.cpu().numpy()
    chol_corr = est_params["chol_corr"].data.cpu().numpy()

    L_covs = [np.diag(sig) @ L for sig, L in zip(sigma, chol_corr)]
    covs = np.array([L @ L.T for L in L_covs])

    if learn_mean:
        means = est_params["mean"].data.cpu().numpy()
        return initial_probs, transition_probs, means, covs

    else:
        return initial_probs, transition_probs, covs


import jax
import jax.numpy as jnp

from jax import lax


@jax.jit
def compute_viterbi_path(x, initial_probs, transition_probs, means, covs):
    """
    Compute the most likely state sequence (i.e., MAP sequence) for the observed data x using the Viterbi algorithm.


    notes:
    # This code is adapted from the dynamax package [1]:
    # https://github.com/probml/dynamax/blob/main/dynamax/hidden_markov_model/inference.py#L461
    # The original code is slightly modified for this project

    references:

    [1] Linderman, Scott & Chang, Peter & Harper-Donnelly, Giles & Kara, Aleyna & Li, Xinglong & Duran-Martin, Gerardo & Murphy, Kevin. (2025). Dynamax: A Python package for probabilistic state space modeling with JAX. Journal of Open Source Software. 10. 7069. 10.21105/joss.07069.
    [2] "Machine Learning: Advanced Topics", K. Murphy, MIT Press 2023. Available at https://probml.github.io/pml-book/book2.html.


    Returns
    -------
    viterbi_path : jnp.array
        The most likely sequence of states as a 1D array.
    """

    def mvn_log_likelihood(x, means, covs):
        # Computes the logpdf of a multivariate normal for each observation given a set of covariance matrices.
        def _mvn_logpdf(x, mean, cov):
            return jss.multivariate_normal.logpdf(x, mean, cov)

        # Vectorize over states for a fixed observation.
        _log_pdf_over_state = jax.vmap(_mvn_logpdf, in_axes=(None, 0, 0))

        # Vectorize over time (observations).
        log_likelihood = jax.vmap(_log_pdf_over_state, in_axes=(0, None, None))
        return log_likelihood(x, means, covs)

    log_likelihoods = mvn_log_likelihood(x, means, covs)  # shape: (num_data, num_states)
    print(log_likelihoods.shape)
    num_data, num_states = log_likelihoods.shape

    # Backward pass: compute best scores and corresponding next states.
    def _backward_pass(best_next_score, t):
        # For time index t, compute scores for transitioning to each state.
        scores = jnp.log(transition_probs) + best_next_score + log_likelihoods[t + 1]
        best_next_state = jnp.argmax(scores, axis=1)
        best_next_score = jnp.max(scores, axis=1)
        return best_next_score, best_next_state

    # Scan backwards over time indices.
    best_second_score, best_next_states = lax.scan(
        _backward_pass, jnp.zeros(num_states), jnp.arange(num_data - 1), reverse=True
    )

    # Forward pass: reconstruct the optimal state path.
    def _forward_pass(state, best_next_state):
        next_state = best_next_state[state]
        return next_state, next_state

    first_state = jnp.argmax(jnp.log(initial_probs) + log_likelihoods[0] + best_second_score)
    _, states = lax.scan(_forward_pass, first_state, best_next_states)
    viterbi_path = jnp.concatenate([jnp.array([first_state]), states])

    return viterbi_path


import jax.scipy.stats as jss


def mvn_log_likelihood(x, means, covs):
    # Computes the logpdf of a multivariate normal for each observation given a set of covariance matrices.
    def _mvn_logpdf(x, mean, cov):
        return jss.multivariate_normal.logpdf(x, mean, cov)

    # Vectorize over states for a fixed observation.
    _log_pdf_over_state = jax.vmap(_mvn_logpdf, in_axes=(None, 0, 0))

    # Vectorize over time (observations).
    log_likelihood = jax.vmap(_log_pdf_over_state, in_axes=(0, None, None))
    return log_likelihood(x, means, covs)


def compute_stationary_distribution(trans_mat):

    """
    Compute stationary distribution

    notes:
    # This code is adapted from the following cite:
    # https://ninavergara2.medium.com/calculating-stationary-distribution-in-python-3001d789cd4b

    """


    eigvals, eigvecs = jnp.linalg.eig(trans_mat.T)
    stationary_dist = np.real(eigvecs[:, np.isclose(eigvals, 1.0)])[:, 0]
    return stationary_dist / np.sum(stationary_dist)  # from eigvec to probs