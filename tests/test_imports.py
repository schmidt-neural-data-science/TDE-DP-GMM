import importlib


def test_top_level_import_exposes_release_version():
    package = importlib.import_module("nds_toolbox")

    assert package.__version__ == "0.1.5"


def test_representative_submodule_imports():
    module_names = [
        "nds_toolbox.preprocessing.features",
        "nds_toolbox.preprocessing.filters",
        "nds_toolbox.analysis.utils",
        "nds_toolbox.sim.bursts.simulator",
        "nds_toolbox.models.hmm_pyro",
        "nds_toolbox.models.dpgmm_numpyro",
    ]

    for module_name in module_names:
        importlib.import_module(module_name)
