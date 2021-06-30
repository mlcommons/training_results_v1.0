# SSD

## Benckmark Information

SSD is
[ssd-resnet34 300x300](https://github.com/mlperf/training/tree/master/single_stage_detector/ssd) benchmark.

## Software

Tensorflow v1.

## Hardware
TPU v4.

## Model
### Publication/Attribution

Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox Detector. In the
Proceedings of the European Conference on Computer Vision (ECCV), 2016.

## Dataset Preparation

Microsoft COCO: COmmon Objects in Context. 2017.

*   [SSD COCO Dataset preparation](https://github.com/tensorflow/tpu/tree/master/models/official/retinanet#preparing-the-coco-dataset)

## Issues with source code

The submitted source code was not checking whether evaluations had completed prior to convergence.
We added file ssd-preview-TF-tpu-v4-128/ssd_main_eval_fix.py with fixes to this problem.
This file replaces/fixes file ssd-preview-TF-tpu-v4-128/ssd_main.py.
In future submissions such cases will be automatically detected by the compliance checker.

## Preview submissions

Preview submissions were run using Google internal infrastructure. 
Contact Peter Mattson (petermattson@google.com) for more details.
