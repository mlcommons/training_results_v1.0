# MLPerf RN50 CNN Training on IPUs

This readme describes how to run the benchmarks.
This code runs with Poplar SDK 2.1.
Install the SDK and the requirement.txt in the implementations folder.
The TF records have to be located at `/localdata/datasets/imagenet-data`.

For running the benchmarks for the different
accelerator sizes (4, 16, 64),
execute:

```
for i in `seq 0 4`; do  ./run_and_time.sh NUM_ACCELERATORS $i; done
```

The result files will be in the respective logging folders of each run.
The parameter configurations can be found in `configs.yml`
and `run_and_time.sh`.