# 1. Problem

This benchmark represents a 3D medical image segmentation task using [2019 Kidney Tumor Segmentation Challenge](https://kits19.grand-challenge.org/) dataset called [KiTS19](https://github.com/neheller/kits19). The task is carried out using a [U-Net3D](https://arxiv.org/pdf/1606.06650.pdf) model variant based on the [No New-Net](https://arxiv.org/pdf/1809.10483.pdf) paper.

## Requirements
* [PyTorch 21.03-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch) (data preprocessing)
* [MXNet 21.03-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-mxnet)
* Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot) (multi-node)
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (single-node)

## Dataset

The data is stored in the [KiTS19 github repository](https://github.com/neheller/kits19).

## Publication/Attribution
Heller, Nicholas and Isensee, Fabian and Maier-Hein, Klaus H and Hou, Xiaoshuai and Xie, Chunmei and Li, Fengyi and Nan, Yang and Mu, Guangrui and Lin, Zhiyong and Han, Miofei and others.
"The state-of-the-art in kidney and kidney tumor segmentation in contrast-enhanced CT imaging: Results of the KiTS19 Challenge".
Medical Image Analysis, 101821, Elsevier (2020).

Heller, Nicholas and Sathianathen, Niranjan and Kalapara, Arveen and Walczak, Edward and Moore, Keenan and Kaluzniak, Heather and Rosenberg, Joel and Blake, Paul and Rengel, Zachary and Oestreich, Makinna and others.
"The kits19 challenge data: 300 kidney tumor cases with clinical context, ct semantic segmentations, and surgical outcomes".
arXiv preprint arXiv:1904.00445 (2019).

# 2. Directions

## Steps to download and verify data

1. Build and run the dataset preprocessing Docker container.
    
    ```bash
    docker build -t preprocessing -f Dockerfile_pyt .
    docker run --ipc=host -it --rm --runtime=nvidia -v DATASET_DIR:/data preprocessing:latest 
    ```
   Where DATASET_DIR is the target directory used to store the dataset after preprocessing.

   
2. Download and preprocess the data

    ```bash
    bash download_dataset.sh 
    ```

## Steps to launch training on a single node

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to
run our container.

### NVIDIA DGX-1/DGX A100 (single node)

Launch configuration and system-specific hyperparameters for the appropriate
single node submission on NVIDIA DGX1 or DGXA100 are in the `config_DGX1*.sh` or `config_DGXA100*.sh`scripts, respectively.

Steps required to launch single node training:

1. Build the container and push to a docker registry:
```
docker build --pull -t <docker/registry>/mlperf-nvidia:image_segmentation-mxnet .
docker push <docker/registry>/mlperf-nvidia:image_segmentation-mxnet
```
2. Launch the training:

```
source config_DGXA100_conv-dali_1x8x4.sh # or config_DGX1_conv-dali_1x8x4.sh or config_DGXA100_conv-dali_1x8x7.sh
CONT="<docker/registry>/mlperf-nvidia:image_segmentation-mxnet" DATASET_DIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```

### Alternative launch with nvidia-docker

When generating results for the official v0.7 submission with one node, the
benchmark was launched onto a cluster managed by a SLURM scheduler.

However, to make it easier to run this benchmark on a wider set of machine
environments, we are providing here an alternate set of launch instructions
that can be run using nvidia-docker. Note that performance or functionality may
vary from the tested SLURM instructions.

```
docker build --pull -t mlperf-nvidia:image_segmentation-mxnet .
source config_DGXA100_conv-dali_1x8x4.sh # or config_DGX1_conv-dali_1x8x4.sh or config_DGXA100_conv-dali_1x8x7.sh
CONT=mlperf-nvidia:image_segmentation-mxnet DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> ./run_with_docker.sh
```


## Steps to launch training on multiple nodes

For multi-node training, we use Slurm with the Pyxis extension, and Slurm's MPI
support to run our container, and correctly configure the environment for
MXNet/Horovod distributed execution.

### NVIDIA DGX A100 (multi node)

Launch configuration and system-specific hyperparameters for the NVIDIA DGX
A100 multi node submissions are in the `config_DGXA100_multi_8x8x24.sh`,
`config_DGXA100_multi_64x8x4x4.sh`, or `config_DGXA100_multi_128x8x2x8.sh`
scripts.
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-2H
multi node submission are in the `config_DGX2_multi_64x16x02x08_bmul.sh`
script.

Steps required to launch multi node training

1. Build the docker container and push to a docker registry
```
docker build --pull -t <docker/registry>/mlperf-nvidia:image_segmentation-mxnet .
docker push <docker/registry>/mlperf-nvidia:image_segmentation-mxnet
```

2. Launch the training
```
source config_DGXA100_multi_8x8x24.sh # or one of the other config_*_multi*.sh scripts
CONT="<docker/registry>/mlperf-nvidia:image_segmentation-mxnet" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```

### Hyperparameter settings

Hyperparameters are recorded in the `config_*.sh` files for each configuration and in `run_and_time.sh`.

 
# 3. Quality

## Quality metric

The quality metric in this benchmark is mean (composite) DICE score for classes 1 (kidney) and 2 (kidney tumor). 
The metric is reported as `mean_dice` in the code.

## Quality target

The target `mean_dice` is 0.908.

## Evaluation frequency

The evaluation schedule is the following:
- for epochs 1 - 999: Do not evaluate
- for epochs >= 1000: Evaluate every 20 epochs

## Evaluation thoroughness

The validation dataset is composed of 42 volumes. They were pre-selected, and their IDs are stored in the `evaluation_cases.txt` file.
A valid score is obtained as an average `mean_dice` score across the whole 42 volumes. Please mind that a multi-worker training in popular frameworks is using so-called samplers to shard the data.
Such samplers tend to shard the data equally across all workers. For convenience, this is achieved by either truncating the dataset, so it is divisible by the number of workers,
or the "missing" data is duplicated using existing samples. This most likely will influence the final score - a valid evaluation is performed on exactly 42 volumes and each volume's score has a weight of 1/42 of the total sum of the scores.
