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

import sys
import os
sys.path.append('/home/ma-user/modelarts-sdk')
os.environ['S3_ENDPOINT'] = 'obs.cn-south-222.ai.pcl.cn'

from modelarts.session import Session
from modelarts.estimator import Estimator
import time
from datetime import datetime
import logging
import moxing as mox


TRAIN_INSTANCE_TYPE='modelarts.vm.cpu.8u'
POOL_ID ='pool55b74902'
REGION_NAME='cn-south-222'
FRAMEWORK_TYPE='Ascend-Powered-Engine'
FRAMEWORK_VERSION='MindSpore-1.1.1-c76-tr5-python3.7-euleros2.8-aarch64'
CODE_DIR='/mlperf/bert/'
BOOT_FILE='/mlperf/bert/modelarts/train_cloud_copy_dataset.py'
OUTPUT_PATH='/mlperf/output_copy_data/copy_data_{}/'

TRAIN_INSTANCE_COUNT=1
JOB_DESCRIPTION='Bert'
PREFIX_JOB_NAME='bert_copy_data'
INPUTS='/mlperf/dataset/en-wiki-20200101/'
ACCESS_KEY='OERLEILJO4NYWNBJUCEO'
SECRET_KEY='kDsH25EkEbHeg90LUfauHS5I4rBzqaoQ8NpUHg2m'
PROJECT_ID='b4034a32b061411aadadcae6868d53a5'

REPEAT_TIME = 1

PARAMS=[]

def create_output_path(path):
    _path = "s3:/" + path
    if mox.file.exists(_path):
        mox.file.remove(_path, recursive=False)
        time.sleep(10)
    mox.file.make_dirs(_path)


def main():    
    session = Session()
   
    #train_instance_type_list = Estimator.get_train_instance_types(session)
    #print(train_instance_type_list)
    if REPEAT_TIME > 1:
        _OUTPUT_PATH = OUTPUT_PATH.format(0)
    else:
        _OUTPUT_PATH = OUTPUT_PATH.format(0)
    
    
    create_output_path(_OUTPUT_PATH)
    estimator = Estimator(modelarts_session=session,
                   framework_type=FRAMEWORK_TYPE,
                   framework_version=FRAMEWORK_VERSION,
                   code_dir=CODE_DIR,
                   boot_file=BOOT_FILE,
                   hyperparameters=PARAMS,
                   output_path=_OUTPUT_PATH,
                   log_url=_OUTPUT_PATH,
                   train_instance_type=TRAIN_INSTANCE_TYPE,
                   train_instance_count=TRAIN_INSTANCE_COUNT,
                   job_description=JOB_DESCRIPTION,
                   pool_id = POOL_ID)

    job_name = PREFIX_JOB_NAME + '_' + datetime.now().strftime("%Y%m%d%H%M%S")   
    job_instance = estimator.fit(job_name=job_name,
                        inputs=INPUTS,
                        wait=False)

    cur_job_id = job_instance.job_id
    cur_version_id = job_instance.version_id
    print("The job start:%s, job_instance.job_id:%s, job_instance.version_id:%s" %(job_name,str(cur_job_id),str(cur_version_id)))

    time.sleep(10)
    for i in range(1, REPEAT_TIME):
        _OUTPUT_PATH = OUTPUT_PATH.format(i)
        create_output_path(_OUTPUT_PATH)
        estimator.output_path = _OUTPUT_PATH
        estimator.log_url = _OUTPUT_PATH
        job_version_instance = estimator.create_job_version(job_id=cur_job_id, pre_version_id=cur_version_id, inputs=INPUTS, wait=False)
        cur_job_id = job_version_instance.job_id                                                   
        cur_version_id = job_version_instance.version_id
        print("The job start:%s, job_instance.job_id:%s, job_instance.version_id:%s" %(job_name,str(cur_job_id),str(cur_version_id)))
        time.sleep(10)


if __name__ == '__main__':
    main()
