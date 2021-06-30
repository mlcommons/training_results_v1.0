##Steps to launch training on multiple nodes

### NVIDIA DGX A100 (multi node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX A100
single node submission are in the `config_DGXA100_8x8x1120.sh` script
and in the `dgx_a100_8x8x1120.json` config file.

To launch the trainining on a single node with a slurm cluster run:
```
source config_DGXA100_8x8x1120.sh
CONT=<docker/registry>/mlperf-nvidia:recommendation_hugectr LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES run.sub
```