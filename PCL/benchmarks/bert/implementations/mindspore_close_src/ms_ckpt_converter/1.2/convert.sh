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

python ./ms_and_tf_checkpoint_transfer_tools.py \
    --tf_ckpt_path="/home/ME/Downloads/mlperf_ckpt/bs64k_32k_ckpt_model.ckpt-28252" \
    --ms_ckpt_path="./bert_large.ckpt" \
    --new_ckpt_path="./ms_bert_large.ckpt" \
    --transfer_option="tf2ms"

