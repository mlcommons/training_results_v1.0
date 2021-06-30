# mlperf-training SSD in mxnet

This is stripped out of [Gluon CV's](https://github.com/dmlc/gluon-cv) Model
Zoo, and then modified to support some of the options we need to
match the SSD model in mlperf (Resnet-34 V1.5 backbone, with NHWC and fp16).

## Running code in Docker

### Launching the container

Use the following command to launch a prebuilt model container:

``
scripts/docker/launch.sh <DATASET_DIR> <RESULTS_DIR>
``

On the first execution of the script, docker will download the model image from NGC container registry.
The contents of this repository will be mounted to the `/workspace/ssd` directory inside the container.
Additionally, `<DATA_DIR>` and `<RESULT_DIR>` directories on the host will be mounted to `/datasets`, `/results` respectively.

### Building the image

This is an optional step and only applies if you want to build the image yourself, NGC container registry includes a prebuilt image.

To build the model image yourself, run the following script:

``
scripts/docker/build.sh
``

Note that the source and target image tags are defined in:

``
scripts/docker/config.sh
``

Make sure to change the target image name if you intended to upload your own image to NGC.


### Downloading the datasets and pretrained weights

Run the following script to download COCO-2017 dataset:

``
scripts/datasets/download_coco2017.sh
``

The compressed dataset will be downloaded to

``
/datasets/downloads
``

And the extracted files will be in:

``
/datasets/coco2017
``


### Download pretrained ResNet34 weights

From your **host**, execute the following script to obtain the ResNet34
pretrained weights:

``
scripts/datasets/get_resnet34_backbone.sh
``

The script will download ResNet34 weights from torchvision (`.pth` format) then
convert it to a `.pickle` file readable by mxnet. The script will automatically
download and run a PyTorch container for the conversion.


### Training the network

Use any of the scripts under

``
scripts/train/*
``

The script names should be self explanatory.

Please note that by default the training scripts expect the pretrained ResNet34
weights to be found at:

``
/datasets/backbones/resnet34-333f7ec4.pickle
``
