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

PARAMS = [
    {"label": "distribute", "value": "true"},
    {"label": "train_with_eval", "value": "true"},
    {"label": "eval_data_dir", "value": "/cache/data/eval"},
    {"label": "data_dir", "value": "/cache/data/train"},
    {"label": "enable_save_ckpt", "value": "false"},
    {"label": "enable_lossscale", "value": "true"},
    {"label": "do_shuffle", "value": "true"},
    {"label": "data_sink_steps", "value": "-1"},
    {"label": "train_steps", "value": "820"},
    {"label": "accumulation_steps", "value": "1"},
    {"label": "load_checkpoint_path", "value": "/home/work/user-job-dir/bert_base_plus/ms_bert_large.ckpt"},
    {"label": "lr_start", "value": "2.5e-3"},
    {"label": "lr_end", "value": "1e-5"},
    {"label": "total_steps", "value": "820"},
    {"label": "batch_size", "value": "16"},
    {"label": "eval_batch_size", "value": "80"},
    {"label": "beta1", "value": "0.81"},
    {"label": "beta2", "value": "0.88"},
    {"label": "weight_decay", "value": "0.0166629"},
    {"label": "warmup_steps", "value": "256"},
    {"label": "start_warmup_steps", "value": "-76"},
    {"label": "eps", "value": "1e-7"},
]