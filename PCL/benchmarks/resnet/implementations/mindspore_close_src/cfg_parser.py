# Copyright 2021 PCL & PKU
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

import yaml
import sys
import os
import re


class ConfigObject:
    def __init__(self, entries):
        for a, b in entries.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [ConfigObject(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, ConfigObject(b) if isinstance(b, dict) else b)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


def parse_yaml(fp):
    with open(fp, 'r') as fd:
        cont = fd.read()
        try:
            y = yaml.load(cont, Loader=yaml.FullLoader)
        except:
            y = yaml.load(cont)
        return y


def merge_args(args, args_yml_fn):
    if os.path.exists(args_yml_fn):
        args_dict = args.__dict__
        args_yml = parse_yaml(args_yml_fn)
        args_dict_merge = dict(args_dict, **args_yml)
        args = ConfigObject(args_dict_merge)
    elif len(args_yml_fn) != 0:
        print('yml file {} is not existed'.format(args_yml_fn))
        exit(0)

    sys_args = sys.argv[1:]
    for arg in sys_args:
        if re.match('^--(.*)=(.*)$', arg):
            arg = arg.replace('--', '')
            key, val = arg.split('=')
            if key not in ['local_rank', 'world_size', 'args_yml_fn']:
                default_value = getattr(args, key)
                setattr(args, key, type(default_value)(val))

    return args
