# Steps to launch training
Submissions assume that you have:

1.  The user has an OBS storage bucket which is located in the same region as your ModelArts service.
2.  The user instance must have permissions to access ModelArts APIs.
3.  The project must have quota to create ModelArts training jobs for the submission.

## Dataset Preparation

1. Download original data from image-net.org.
2. Organize the data as follows:

```
data/imagenet/
|_ train/
|  |_ n01440764
|  |  |_ n01440764_10026.JPEG
|  |  |_ ...
|  |_ ...
|  |
|  |_ n15075141
|     |_ ...
|     |_ n15075141_9993.JPEG
|_ val/
|  |_ n01440764
|  |  |_ ILSVRC2012_val_00000293.JPEG
|  |  |_ ...
|  |_ ...
|  |
|  |_ n15075141
|     |_ ...
|     |_ ILSVRC2012_val_00003181.JPEG
```
3.  Upload the data to OBS.

## Run

1.  Upload the benchmark scripts to OBS.

2.  Once the dataset and scripts are ready, simply launch `train.py` as a training job on ModelArts with the number of nodes.

3.  Get result logs on OBS path specified in the training job.