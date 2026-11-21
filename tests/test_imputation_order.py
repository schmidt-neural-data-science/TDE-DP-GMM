import numpy as np
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score, matthews_corrcoef

from nds_toolbox.utils.helper import return_performance_info


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
