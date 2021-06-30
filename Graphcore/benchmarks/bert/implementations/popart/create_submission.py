# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import argparse
import numpy as np
import subprocess
import os
import re

# Specify which pod to run
parser = argparse.ArgumentParser("Config Parser", add_help=False)
parser.add_argument("--pod", type=int, choices=[16, 64], default=16)
parser.add_argument("--submission-division", type=str, choices=["open", "closed"], default="closed")
parser.add_argument("--start-index", type=int, default=0)
parser.add_argument("--end-index", type=int, default=10)
args = parser.parse_args()

# Each submission consist of 10 runs
for result_index in range(args.start_index, args.end_index):
    command = f"python bert.py --config=configs/mk2/pod{args.pod}-{args.submission_division}.json --seed {result_index + 42}"
    options = f"--submission-run-index={result_index}"

    # Launch the run
    with open(f"internal_log_{result_index}", "w+") as f:
        # Clear the cache
        # subprocess.call(['sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"'], stdout=f, stderr=f, shell=True)

        # Run training
        subprocess.call([command + " " + options], stdout=f, stderr=f, shell=True)
