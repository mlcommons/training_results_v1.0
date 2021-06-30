
# Instructions
Follow setup instructions in the general implementation [README](../implementations/popart/README.md) to get started with Bert for IPU

To launch individual runs:
  ```
  python3 ./bert.py --config=configs/mk2/pod16-closed.json
  ```
The first run will produce and store the executable. 

Afterwards, run 10 times to generate Mlperf submission logs and reproduce results:
```
sudo ./run_and_time.sh
```
Submission logs will be recorded to `./results/bert/result_*.txt`
