### Steps to download data
```
DATADIR=<path/to/data/dir> ./init_datasets.sh
```

## Steps to launch training

### FUJITSU PRIMERGY GX2460M1 (single node)
Launch configuration and system-specific hyperparameters for the FUJITSU PRIMERGY GX2460M1
single node submission are in the `config_PG.sh` script.

Steps required to launch single node training on FUJITSU PRIMERGY GX2460M1:

```
BENCHMARK=<benchmark_test> ./setup.sh
BENCHMARK=<benchmark_test> DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> ./run_and_time.sh
```
