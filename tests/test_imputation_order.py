import numpy as np
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score, matthews_corrcoef

from nds_toolbox.analysis.utils import imputing_mode, imputing_mode_batched
from nds_toolbox.utils.helper import return_performance_info


def test_imputation_preserves_runs_and_votes_across_batch_boundaries():
    # The first expanded window reaches both label 8 and a majority of label 2.
    states = np.array([-9] * 7 + [8, 2, 2, 2, -9, 4, 4, -9])
    expected = imputing_mode(states, min_samples=1, window_size=12, missing_value=-9)
    for batch_size in (1, 2, 5, 64):
        result = imputing_mode_batched(states, min_samples=1, window_size=12,
                                       missing_value=-9, batch_size=batch_size)
        np.testing.assert_array_equal(result, expected)

    # The run of 5s survives, and its votes propagate through two short runs.
    states = [5, 5, 5, 5, 8, 7, 9, 9, 9, 9]
    expected = imputing_mode(states, min_samples=4, window_size=2)
    for batch_size in (1, 3, 4, 64):
        result = imputing_mode_batched(states, min_samples=4, window_size=2,
                                       batch_size=batch_size)
        np.testing.assert_array_equal(result, expected)


def test_imputation_returns_original_when_no_run_survives():
    for states in ([], [-1, -1, -1], [1, 1, 0, 0, 1]):
        result = imputing_mode_batched(states, min_samples=3, batch_size=2)
        np.testing.assert_array_equal(result, states)


def test_imputation_votes_in_matched_order_and_scores_the_final_labels():
    decoded = np.array([1, 1, 2, 0, 0, 0, 2, 2, 2, 1, 1, 1])
    truth = np.array([0, 0, 1, 3, 3, 3, 1, 1, 1, 0, 0, 0])
    config = dict(
        method="HMM",
        signal=np.zeros(len(truth)),
        fs=250,
        true_states=truth,
        est_states=decoded,
        min_samples=2,
        num_states=3,
        num_emb=1,
        est_params={
            "initial_probs": np.full(3, 1 / 3),
            "transition_probs": np.full((3, 3), 1 / 3),
            "means": np.arange(3)[:, None],
            "covs": np.ones((3, 1, 1)),
        },
        loss=0,
        metrics=("ari", "mcc", "kappa"),
    )
    raw = return_performance_info(**config)
    imputed = return_performance_info(**config, imputing_spurious_states=True)

    # At index 2 the neighbors tie. Their parameter indices are 1 and 0,
    # while their indices after matching are 0 and 2, respectively.
    np.testing.assert_array_equal(raw["order"], [1, 2, 0])
    np.testing.assert_array_equal(raw["state_labels"], [0, 1, 3])
    np.testing.assert_array_equal(imputed["order"], raw["order"])
    np.testing.assert_array_equal(imputed["state_labels"], raw["state_labels"])
    np.testing.assert_array_equal(imputed["est_params"]["means"], raw["est_params"]["means"])
    expected = np.array([0, 0, 0, 2, 2, 2, 1, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(imputed["est_states"], expected)

    matched = raw["state_labels"][expected]
    np.testing.assert_array_equal(imputed["matched_est_states"], matched)
    assert imputed["ari"] == adjusted_rand_score(truth, matched)
    assert imputed["mcc"] == matthews_corrcoef(truth, matched)
    assert imputed["kappa"] == cohen_kappa_score(truth, matched)
