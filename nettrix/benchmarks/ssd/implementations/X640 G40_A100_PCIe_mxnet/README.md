# 1. Problem

Single Stage Object Detection.

## Requirements
* [MXNet 21.05-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-mxnet)
* Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot) (multi-node)
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (single-node)

# 2. Directions

### Steps to download data

First download the coco 2017 dataset.  This can be done using the instructions
in the Mlperf reference single_stage_detector implementation:

```
cd reference/single_stage_detector/
source download_dataset.sh
```

Next convert the required reference ResNet-34 pretrained backbone weights
(which, in the reference, come from TorchVision) into a format readable by
non-pytorch frameworks.  From the directory containing this README run:

```
source scripts/get_resnet34_backbone.sh
```

## Steps to launch training on a single node

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to
run our container.

### Nettrix X640\ G40_A100_PCIe (single node)

Launch configuration and system-specific hyperparameters for the appropriate
NVIDIA DGX single node submission are in the `config_X640\ G40_A100_PCIe.sh` script respectively.

Steps required to launch single node training:

1. Build the container and push to a docker registry:
```
docker build --pull -t <docker/registry>/mlperf-nvidia:single_stage_detector-mxnet .
docker push <docker/registry>/mlperf-nvidia:single_stage_detector-mxnet
```
2. Launch the training:

```
source config_X640\ G40_A100_PCIe_.sh
CONT="<docker/registry>/mlperf-nvidia:single_stage_detector-mxnet DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME run_X640\ G40_A100_PCIe_.sub
```

### Alternative launch with nvidia-docker

When generating results for the official v0.7 submission with one node, the
benchmark was launched onto a cluster managed by a SLURM scheduler. The
instructions in [NVIDIA DGX-1/DGX-2/DGX A100 (single
node)](#nvidia-dgx-1dgx-2dgx-a100-single-node) explain how that is done.

However, to make it easier to run this benchmark on a wider set of machine
environments, we are providing here an alternate set of launch instructions
that can be run using nvidia-docker. Note that performance or functionality may
vary from the tested SLURM instructions.

```
docker build --pull -t mlperf-nvidia:single_stage_dector-mxnet .
source config_X640\ G40_A100_PCIe_.sh
CONT=mlperf-nvidia:single_stage_detector-mxnet DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> ./run_with_docker_X640\ G40_A100_PCIe_.sh
```

## Steps to launch training on multiple nodes

For multi-node training, we use Slurm with the Pyxis extension, and Slurm's MPI
support to run our container, and correctly configure the environment for
MXNet/Horovod distributed execution.

### Hyperparameter settings

Hyperparameters are recorded in the `config_*.sh` files for each configuration and in `run_and_time_X640\ G40_A100_PCIe_.sh`.

# 3. Dataset/Environment
### Publiction/Attribution.
Microsoft COCO: COmmon Objects in Context. 2017.

### Training and test data separation
Train on 2017 COCO train data set, compute mAP on 2017 COCO val data set.

# 4. Model.
### Publication/Attribution
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox Detector. In the Proceedings of the European Conference on Computer Vision (ECCV), 2016.

Backbone is ResNet34 pretrained on ImageNet (from torchvision).

# 5. Quality.
### Quality metric
Metric is COCO box mAP (averaged over IoU of 0.5:0.95), computed over 2017 COCO val data.

### Quality target
mAP of 0.23

### Evaluation frequency
The model is evaluated at epochs 40, 50, 55, and then every 5th epoch.

### Evaluation thoroughness
All the images in COCO 2017 val data set.
