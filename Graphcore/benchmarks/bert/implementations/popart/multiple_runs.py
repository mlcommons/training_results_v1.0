import numpy as np
import subprocess
import os
import re

eval_accuracies = {}

for i, beta1 in enumerate([0.55, 0.6, 0.7, 0.75]):
    command = "python bert.py --config=configs/mk2/pod64.json"
    options = f"--enable-tensorboard --beta1={beta1}"

    # Launch the run
    with open(f"log_{i}", "w+") as f:
        subprocess.call([command + " " + options], stdout=f, stderr=f, shell=True)
    print(f"Done with run for beta1 = {beta1}")

    # Extract accuracy values from the run
    with open(f"log_{i}", "r") as f:
        txt = f.readlines()
        txt = "\n".join(txt)
    a = list(map(float, re.findall("Eval accuracy: ([\d\.]+)", txt)))
    eval_accuracies[beta1] = a


# Write the dictionary of accuracies for each hparam
with open("results.txt", "w+") as f:
    f.write(str(eval_accuracies))
