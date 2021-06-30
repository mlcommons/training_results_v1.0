## Wikipedia pre-training data

Follow the Mlcommons' reference implementation instructions to construct the training and eval datasets

## Pack sequences to reduce padding:

First convert the tfrecords to a binary format using `bert_data/record_to_binary.py`
```
python3 record_to_binary --tf-record-glob=cleanup_scripts/tf_records/part** --output-path=binarized_data
```
Then pack the sequence data using `bert/pack_pretraining_data`:
```
python3 pack_pretraining_data.py --input-glob="bert_data/binarized_data/part_*" --output-dir="data/packed_pretraining_data"`
```
The same steps should also be repeated for the eval dataset.
The wikipedia dataset is now ready to be used in the Graphcore BERT model.