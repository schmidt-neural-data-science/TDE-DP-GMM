# NDS Toolbox

Neural data science utilities for simulating bursty signals, preprocessing time
series, fitting probabilistic state models, and summarizing burst/state
structure.

The package is distributed as `nds-toolbox` and imported as `nds_toolbox`.

## Installation

Install the released package from PyPI:

```bash
python -m pip install nds-toolbox
```

For development from this repository:

```bash
python -m pip install -e ".[dev]"
```

## Quick Start

```python
import nds_toolbox

print(nds_toolbox.__version__)
```

Example submodule imports:

```python
from nds_toolbox.preprocessing.features import compute_tde, trim_data
from nds_toolbox.sim.bursts.simulator import simulate_bursty_signal
from nds_toolbox.models.hmm_pyro import fit_HMM, compute_viterbi_path
from nds_toolbox.models.dpgmm_numpyro import fit_DPGMM
```

## Modules

- `nds_toolbox.sim`: bursty signal simulation utilities.
- `nds_toolbox.preprocessing`: feature extraction and filtering utilities.
- `nds_toolbox.models`: Pyro and NumPyro probabilistic state models.
- `nds_toolbox.analysis`: burst and state summary utilities.
- `nds_toolbox.utils`: helper functions for decoding and model comparison.

