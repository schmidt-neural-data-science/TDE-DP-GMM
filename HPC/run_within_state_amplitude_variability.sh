#!/usr/bin/env bash
#SBATCH -p cpu
#SBATCH -A smidtrc8_0001
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH -t 7-00:00:00
#SBATCH --hint=nomultithread
#SBATCH -J tde_dp_gmm_within_state_amplitude_variability
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err


set -euo pipefail

echo "JobID: ${SLURM_JOB_ID:-unset}"
echo "Host: $(hostname)"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"
echo "SLURM_JOB_CPUS_PER_NODE: ${SLURM_JOB_CPUS_PER_NODE:-unset}"

ENV_NAME="${ENV_NAME:-pymc_env}"

# Conda activation can reference unset variables.
set +u
source "/home/${USER}/miniforge3/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

cd "${HOME}/projects/TDE-DP-GMM/scripts"

# Limit supported BLAS/OpenMP thread pools in each worker.
# These variables do not impose a universal JAX thread limit.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Python must read this value, for example:
# inner_n_jobs = int(os.environ.get("N_PROCS", "10"))
# Keep the outer condition loop sequential.
export N_PROCS=10

echo "Conda environment: ${ENV_NAME}"
echo "Inner workers requested: ${N_PROCS}"
python -V
command -v python

# Use the toolbox from this checkout and fail early if it lacks metrics.
export PYTHONPATH="${PWD}/..${PYTHONPATH:+:${PYTHONPATH}}"
python -c 'import inspect; from nds_toolbox.utils.helper import compare_decoding_performance as f; print(inspect.getfile(f)); assert "metrics" in inspect.signature(f).parameters'

srun python -u within_state_amplitude_variability.py
