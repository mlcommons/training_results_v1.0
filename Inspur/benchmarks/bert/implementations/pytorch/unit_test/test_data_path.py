# NVIDIA

import os

expected_data_path = '/workspace/unit_test_data'

def data_path_found():
  return os.path.isdir(expected_data_path)

def get_path():
  if data_path_found():
    return expected_data_path
  else:
    raise ValueError('Unit test data not found - missing or not mounted correctly.')
