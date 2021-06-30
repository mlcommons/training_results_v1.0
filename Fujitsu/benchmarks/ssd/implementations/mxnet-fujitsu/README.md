# 1. Problem

Single Stage Object Detection.

## Requirements
* [MXNet 21.04-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-mxnet)
* Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot) (multi-node)
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (single-node)

# 2. Directions

### Steps to download data

First download the coco 2017 dataset.  This can be done using the instructions
in the Mlperf reference single_stage_detector implementation:

```
cd ../gx_A100x4_mxnet_NGC_21.04
DATADIR=<path/to/data/dir> ./init_datasets.sh
```

Next convert the required reference ResNet-34 pretrained backbone weights
(which, in the reference, come from TorchVision) into a format readable by
non-pytorch frameworks.  From the directory containing this README run:

```
cd ../gx_A100x4_mxnet_NGC_21.04
source download_resnet34_backbone.sh
```

## Steps to launch training on a single node

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to
run our container.

### Fujitsu PRIMERGY GX2460 M1 (single node)

Launch configuration and system-specific hyperparameters for the appropriate
PRIMERGY GX single node submission are in the `config_PG.sh`.

Steps required to launch single node training:

1. Build the container and push to a docker registry:
```
docker build --pull -t <docker/registry>/mlperf-fujitsu:single_stage_detector .
docker push <docker/registry>/mlperf-fujitsu:single_stage_detector
```
2. Launch the training:

```
source config_PG.sh
CONT="<docker/registry>/mlperf-fujitsu:single_stage_detector" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $PGNNODES -t $WALLTIME run.sub
```

### Alternative launch with nvidia-docker

```
docker build --pull -t mlperf-fujitsu:single_stage_dector .
source config_PG.sh
CONT="mlperf-fujitsu:single_stage_detector" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> ./run_with_docker.sh
```
### Hyperparameter settings

Hyperparameters are recorded in the `config_*.sh` files for each configuration and in `run_and_time.sh`.

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
