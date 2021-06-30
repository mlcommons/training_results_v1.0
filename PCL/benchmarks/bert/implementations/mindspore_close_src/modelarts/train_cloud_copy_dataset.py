# Copyright 2020 PCL & PKU
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""train_imagenet."""
import os
import argparse
import random
import numpy as np
import moxing as mox
import time
import sys
import multiprocessing
from datetime import datetime

import os
import time
import glob
from pathlib import Path

mox.file.set_auth(is_secure=False)

data_name = "en-wiki-20200101"
local_cache_path = "/cache_mlperf"
local_data_path = os.path.join(local_cache_path, data_name)

device_id = int(os.getenv('DEVICE_ID'))   # 0 ~ 7
local_rank = int(os.getenv('RANK_ID'))    # local_rank
device_num = int(os.getenv('RANK_SIZE'))  # device_num

exec_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
project_path = os.path.join(exec_path, '..')
sys.path.insert(0, project_path)

def sync_dataset(data_url):
    import moxing as mox
    import time
    start_t = time.time()
    sync_lock = "/tmp/copy_sync.lock"
    if device_id % min(device_num, 8) == 0 and not os.path.exists(sync_lock):

        if not os.path.exists(local_data_path):
            os.system('sudo mkdir {}'.format(local_data_path))
            os.system('sudo chmod -R 777 {}'.format(local_data_path))
        mox.file.copy_parallel(data_url, local_data_path)

        print("===finish download datasets===")
        try:
            os.mknod(sync_lock)
        except:
            pass

        print("===save flag===")
    while True:
        if os.path.exists(sync_lock):
            break
        time.sleep(1)

    print('--------------------------------------------')
    print('copy data completed!')

    end_t = time.time()
    print('copy cost time {:.2f} sec'.format(end_t-start_t))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bert copy data')
    parser.add_argument("--data_url", type=str, default="", help="dataset url")
    parser.add_argument("--train_url", type=str, default="", help="training url")
    args, unknown_args = parser.parse_known_args()
    print(args.data_url)
    print(args.train_url)

    def custom_print(s):
        print("-------------------------------------------------------------------------------------------------")
        print(s)

    if not os.path.exists(local_cache_path):
        custom_print("/cache_mlperf/en-wiki-20200101 does not exist, create it")
        os.system('sudo mkdir {}'.format(local_cache_path))
        os.system('sudo chmod -R 777 {}'.format(local_cache_path))

    def check_data_exist(path):
        if not os.path.isdir(path):
            return False

        dir_list = os.listdir(path)
        if "eval" not in dir_list:
            print("eval dir lost")
            return False
        if "train" not in dir_list:
            print("train dir lost")
            return False

        train_count = len(os.listdir(os.path.join(path, "train")))
        if train_count != 500:
            print("train file lost, found: {}".format(train_count))
            print("Train file found: {}".format(os.listdir(os.path.join(path, "train"))))
            return False
        eval_count = len(os.listdir(os.path.join(path, "eval")))
        if eval_count != 1:
            print("eval file lost, found: {}".format(eval_count))
            print("Eval file found: {}".format(os.listdir(os.path.join(path, "eval"))))
            return False

        return True

    if not check_data_exist(local_data_path):
        sync_dataset(args.data_url)
    else:
        print("Data cache exist")

    custom_print("after copy data:")
    os.system("ls -alt {}".format(local_data_path))

    custom_print("after copy data:")
    os.system("cd {}/train; ls -lR | grep ^- | wc -l".format(local_data_path))
    os.system("cd {}/eval; ls -lR | grep ^- | wc -l".format(local_data_path))
