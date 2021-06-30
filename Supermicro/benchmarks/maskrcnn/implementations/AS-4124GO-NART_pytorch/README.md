# 1. Problem 
This benchmark uses Mask R-CNN for object detection.

## Requirements

* [PyTorch 21.05-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch)
* Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot)
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

# 2. Directions

### Steps to download and verify data
The Mask R-CNN script operates on COCO, a large-scale object detection, segmentation, and captioning dataset.
To download and verify the dataset use following scripts:
   
    cd dataset_scripts
    ./download_dataset.sh
    ./verify_dataset.sh

This should return `PASSED`. 
To extract the dataset use:
   
    DATASET_DIR=<path/to/data/dir> ./extract_dataset.sh

Mask R-CNN uses pre-trained ResNet50 as a backbone. 
To download and verify the RN50 weights use:
 
    DATASET_DIR=<path/to/data/dir> ./download_weights.sh 

Make sure <path/to/data/dir> exists and is writable.

To speed up loading of coco annotations during training, the annotations can be pickled since unpickling is faster than loading a json.

    python pickle_coco_annotations.py --root <path/to/detectron2/dataset/dir> --ann_file <path/to/coco/annotation/file> --pickle_output_file <path/to/pickled/output/file>

Then go back to main maskrcnn repo:

    cd .. 

### Hyperparameter settings

Hyperparameters are recorded in the `config_*.sh` files for each configuration and in `run_and_time.sh`.

# 3. Dataset/Environment
### Publiction/Attribution.
Microsoft COCO: COmmon Objects in Context. 2017.

### Training and test data separation
Train on 2017 COCO train data set, compute mAP on 2017 COCO val data set.


# 4. Model
### Publication/Attribution

We use a version of Mask R-CNN with a ResNet50 backbone. See the following papers for more background:

[1] [Mask R-CNN](https://arxiv.org/abs/1703.06870) by Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick, Mar 2017.

[2] [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.


### Structure & Loss
Refer to [Mask R-CNN](https://arxiv.org/abs/1703.06870) for the layer structure and loss function.


### Weight and bias initialization
The ResNet50 base must be loaded from the provided weights. They may be quantized.


### Optimizer
We use a SGD Momentum based optimizer with weight decay of 0.0001 and momentum of 0.9.


# 5. Quality
### Quality metric
As Mask R-CNN can provide both boxes and masks, we evaluate on both box and mask mAP.

### Quality target
Box mAP of 0.377, mask mAP of 0.339 

### Evaluation frequency
Once per epoch, 118k.

### Evaluation thoroughness
Evaluate over the entire validation set. Use the [NVIDIA COCO API](https://github.com/NVIDIA/cocoapi/) to compute mAP.
