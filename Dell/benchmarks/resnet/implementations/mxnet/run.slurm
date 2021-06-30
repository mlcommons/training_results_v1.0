#!/bin/bash
#SBATCH --job-name image_classification
#SBATCH -N 2                     # number of nodes
#SBATCH -n 16                     # total number of processes
#SBATCH -w node070,node071
##SBATCH --ntasks-per-node 8      # tasks per node
#SBATCH -t 12:00:00             # wall time
#SBATCH --job-name image_classification
##SBATCH --exclusive             # exclusive node access
##SBATCH --mem=0                 # all mem avail
##SBATCH -p dedicateq
##SBATCH --gres=gpu:4

#source config_2xDSS8440x8A100-PCIE-40GB.sh
source config_${SLURM_JOB_NUM_NODES}xDSS8440x8A100-PCIE-40GB.sh
set -euxo pipefail
#CONT="resnet_20210427.mxnet.sif"
CONT="resnet_20210517.mxnet.sif"
# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"

# Vars with defaults
: "${NEXP:=5}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
#: "${DATADIR:=/mnt/data/ilsvrc12_passthrough}"
: "${DATADIR:=/dev/shm/ilsvrc12_passthrough}"
: "${LOGDIR:=/home/frank/results/resnet-${SLURM_JOB_NUM_NODES}nodes}"
: "${COPY_DATASET:=}"
: "${API_LOG_DIR:=./api_logs}" # apiLog.sh output dir

echo $COPY_DATASET

if [ ! -z $COPY_DATASET ]; then
  readonly copy_datadir=$COPY_DATASET
  srun --ntasks-per-node=1 mkdir -p "${DATADIR}"
  srun --ntasks-per-node=1 ${CODEDIR}/copy-data.sh "${copy_datadir}" "${DATADIR}"
  srun --ntasks-per-node=1 bash -c "ls ${DATADIR}"
fi

# Other vars
readonly _seed_override=${SEED:-}
readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
readonly _cont_name=image_classification
_cont_mounts="${DATADIR}:/data,${LOGDIR}:/results"
if [ "${API_LOGGING:-}" -eq 1 ]; then
    _cont_mounts="${_cont_mounts},${API_LOG_DIR}:/logs"
fi

# MLPerf vars
MLPERF_HOST_OS=$(srun -N1 -n1 bash <<EOF
    source /etc/os-release
#    source /etc/dgx-release || true
#    echo "\${PRETTY_NAME} / \${DGX_PRETTY_NAME:-???} \${DGX_OTA_VERSION:-\${DGX_SWBUILD_VERSION:-???}}"
    echo "\${PRETTY_NAME}"
EOF
)
export MLPERF_HOST_OS

module load shared openmpi/4.1.1rc1 ucx/1.10.0 singularity

# Setup directories
mkdir -p "${LOGDIR}"
srun --ntasks="${SLURM_JOB_NUM_NODES}" mkdir -p "${LOGDIR}"

# Setup container
#srun --ntasks="${SLURM_JOB_NUM_NODES}" --container-image="${CONT}" --container-name="${_cont_name}" true

set -x 
# Run experiments
for _experiment_index in $(seq 1 "${NEXP}"); do
    (
    echo "Beginning trial ${_experiment_index} of ${NEXP}"
	
	hosts=$(scontrol show hostname |tr "\n" " ")

	echo "hosts=$hosts"
	#for node_id in `seq 0 $(($NUM_NODES-1))`; do
	for node in $hosts; do
#	for node in node071 node072; do

        # Print system info
        srun -N 1 -n 1 -w $node mpirun --allow-run-as-root -np 1 singularity exec -B $PWD:/workspace/image_classification --pwd /workspace/image_classification $CONT python -c "
import mlperf_log_utils
from mlperf_logging.mllog import constants
mlperf_log_utils.mlperf_submission_log(constants.RESNET)"

        # Clear caches
        if [ "${CLEAR_CACHES}" -eq 1 ]; then
            srun -N 1 -n 1 -w $node bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"
            srun -N 1 -n 1 -w $node mpirun --allow-run-as-root -np 1 singularity exec -B $PWD:/workspace/image_classification --pwd /workspace/image_classification $CONT python -c "
from mlperf_logging.mllog import constants
from mlperf_log_utils import mx_resnet_print_event
mx_resnet_print_event(key=constants.CACHE_CLEAR, val=True)"
        fi

	done

        # Run experiment
        export SEED=${_seed_override:-$RANDOM}
#mpirun --allow-run-as-root --bind-to none -report-bindings -np $SLURM_NTASKS singularity exec --nv -B $PWD:/workspace/image_classification --pwd /workspace/image_classification $CONT ./mpi_hello_world
		#MPIRUN="mpirun --allow-run-as-root --bind-to none -npernode 8 -mca pml ucx -mca btl ^uct -x UCX_NET_DEVICES=mlx5_0:1  -np $SLURM_NTASKS"
		MPIRUN="mpirun --allow-run-as-root --bind-to none -report-bindings -np $SLURM_NTASKS"
        #srun --kill-on-bad-exit=0 --mpi=pmix --ntasks="$(( SLURM_JOB_NUM_NODES * DGXNGPU ))" --ntasks-per-node="${DGXNGPU}" \
        #    --container-name="${_cont_name}" --container-mounts="${_cont_mounts}" \
        #    ./run_and_time.sh
        ${MPIRUN} singularity exec --nv -B $DATADIR:/data -B $PWD:/workspace/image_classification \
		--pwd /workspace/image_classification \
		$CONT bash ./run_and_time_multi.sh
		) |& tee "${_logfile_base}_${_experiment_index}.log"
done
