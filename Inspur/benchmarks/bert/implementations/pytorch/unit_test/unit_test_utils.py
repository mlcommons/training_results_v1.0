# NVIDIA

import numpy as np

# Fractions of max absolute difference
bins_relative = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]

def max_abs_diff_binning(input_a, input_b):
  abs_diff = np.abs(input_a - input_b)
  max_abs_diff = np.max(abs_diff)
  max_idx = np.argmax(abs_diff)

  counts, bins_absolute = np.histogram(abs_diff, np.array(bins_relative) * max_abs_diff)
  return counts, bins_absolute, bins_relative, max_idx

def pyt_tf_mapping(pyt_state_dict, add_prefix=''):
  pyt_strings = [x for x in pyt_state_dict.keys()]
  converted_strings = [add_prefix + x.replace('.', '/').replace('weight', 'kernel') for x in pyt_strings]
  
  for idx, item in enumerate(converted_strings):
    if 'LayerNorm' in item:
      item = item.replace('kernel', 'gamma').replace('bias', 'beta')
    elif 'embedding' in item:
      item = item.replace('/kernel', '')
    
    if 'layer/' in item:
      item = item.replace('layer/', 'layer_')

    if 'cls/' in item and 'decoder' not in item and 'dense' not in item:
      item = item.replace('bias', 'output_bias')
      item = item.replace('kernel', 'output_weights')

    if 'decoder' in item:
      item = item.replace('predictions/decoder', 'predictions/transform/dense')
      #item = item.replace('predictions/decoder', 'seq_relationship')
    # Add additional rules here

    converted_strings[idx] = item

  return dict(zip(pyt_state_dict.keys(), converted_strings))
