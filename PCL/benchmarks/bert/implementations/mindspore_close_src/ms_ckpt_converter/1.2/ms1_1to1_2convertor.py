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

import os
import argparse

from mindspore.train.serialization import load_checkpoint, save_checkpoint

def convert_ckpt():
    parser = argparse.ArgumentParser(description="convertor")
    parser.add_argument("--ckpt_path", type=str, default="", help="old ckpt path")
    parser.add_argument("--new_ckpt_path", type=str, default="", help="new ckpt path")

    args_opt = parser.parse_args()

    param_data_list = []
    param_dict = load_checkpoint(args_opt.ckpt_path)
    convert_map = {
        "bert.bert.bert_embedding_postprocessor.embedding_table": "bert.bert.bert_embedding_postprocessor.token_type_embedding.embedding_table",
        "bert.bert.bert_embedding_postprocessor.full_position_embeddings": "bert.bert.bert_embedding_postprocessor.full_position_embedding.embedding_table"
    }

    for (key, value) in param_dict.items():
        if key in convert_map:
            param_data_list.append({"name": convert_map[key], "data": value.data})
        else:
            param_data_list.append({"name": key, "data": value.data})
    save_checkpoint(param_data_list, args_opt.new_ckpt_path)

if __name__ == "__main__":
    convert_ckpt()