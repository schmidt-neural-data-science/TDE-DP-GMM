""" Neural Data Science toolbox - public API (install name: nds_toolbox)."""
from importlib import metadata as _meta





"""
# --- burst‑HMM shortcuts ----------------------------------------
from .models.hmm_pyro import (
    fit_HMM,
    fit_HMMs,
    extract_params,
    compute_viterbi_path,
)

# --- preprocessing ---------------------------------------------

from .preprocessing.features import (
    compute_tde,
    trim_data,
    amplitude_envelope,
)


from .preprocessing.filters import (

    gaussian_bandpass_filter,
)



# --- analysis  --------------------------------------------------
from .analysis.burst_analysis import (
    thresholding_bursts,
    optimize_threshold_params,
    get_burst_rate,
    get_power,
    get_peak_frequency,
    get_lifetime,
    get_fractional_occupancy,
)


# --- utils  --------------------------------------------------
from .utils.helper import (
    match_states
)
"""

__version__ = _meta.version(__name__) if _meta else "0.0.1"


"""
Patch	0.1.1 → 0.1.2	Bug fix, typo in docs, tiny internal refactor. No new features, no breaking API.
Minor	0.1.2 → 0.2.0	Back‑compatible new features (new detector, extra argument with default, speedups).
Major	0.2.3 → 1.0.0	Breaking changes—renamed functions, different return shapes, removed APIs.
"""