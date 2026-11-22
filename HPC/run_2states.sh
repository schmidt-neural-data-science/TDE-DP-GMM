#!/usr/bin/env bash
#SBATCH -p cpu
#SBATCH -A smidtrc8_0001
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH -t 7-00:00:00
#SBATCH --hint=nomultithread
#SBATCH -J tde_dp_gmm_2states
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Submit from the project root (create logs/ first):
# array_id=$(sbatch --parsable --array=0-29 HPC/run_2states.sh)
# array_id=${array_id%%;*}
# sbatch --dependency="afterok:${array_id}" --cpus-per-task=1 HPC/run_2states.sh --merge --run-id "${array_id}"

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

driver=run_model/2_states_scenario.py
if [[ ! -f "$driver" ]]; then
    driver=2_states_scenario.py
fi
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    set -- --part-id "${SLURM_ARRAY_TASK_ID}" --run-id "${SLURM_ARRAY_JOB_ID}" "$@"
fi
for arg in "$@"; do
    if [[ "$arg" == "--merge" ]]; then
        exec python -u "$driver" "$@"
    fi
done
srun python -u "$driver" "$@"
