# DLRM

## Benckmark Information

DLRM (Deep Learning Recommendation Model) is a recommendation model originally
published by Facebook. The bencmark applies this model to the Criteo Terabyte
dataset.

## Software

Tensorflow v1.

## Hardware
TPU v4.

## Model: DLRM
### Publication/Attribution

[Deep Learning Recommendation Model for Personalization and Recommendation Systems]
(arxiv.org/abs/1906.00091). 2019.

## Dataset Preparation:
This benchmark uses the [Criteo
Terabyte](https://labs.criteo.com/2013/12/download-terabyte-click-logs/)
dataset.  Instructions for pre-processing the data:

1. Follow the link to download the raw data files day_0.gz, ...,day_23.gz and
   unzip them.

2. Run the criteo_preprocess.py script against the data. This script relies on
   TensorFlow Transform and an Apache Beam runner. Run once to generate
   vocabularies, then again to process the data. Sample invocation:

   ```shell
   python criteo_preprocess.py --input_path=<> --output_path=<> --temp_dir=<>
   --vocab_gen_mode=true
   ```

   ```shell
   python criteo_preprocess.py --input_path=<> --output_path=<> --temp_dir=<>
   --vocab_gen_mode=false
   ```

3. Run the criteo_batched.py script against the processed data:

   ```shell
   python criteo_batched.py --batch_size=<desired_core_batch> --input_path=<>
   --output_path=<>
   ```


## Research submissions

Preview submissions were run using Google internal infrastructure.
Contact Peter Mattson (petermattson@google.com) for more details.
