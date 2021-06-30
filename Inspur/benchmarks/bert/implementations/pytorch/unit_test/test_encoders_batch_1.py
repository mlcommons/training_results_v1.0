# NVIDIA

import numpy as np
import pickle
import unittest

from parameterized import parameterized, parameterized_class

import test_data_path
import unit_test_utils
import global_vars as gv

from modeling import BertConfig, BertLayer

import torch
from torch import nn

class TestEncoders(unittest.TestCase):
  @classmethod
  def setUpClass(self):
    self.device = 'cuda:0'
    self.data_path = test_data_path.get_path() + '/gold/batch_1/'
    self.config_path = '/workspace/phase1/bert_config.json'
    self.bert_config = BertConfig(vocab_size_or_config_json_file=self.config_path)
    
    # Additional config options
    self.bert_config.fused_mha = False
    self.bert_config.fused_gelu_bias = True
    self.bert_config.dense_seq_output = True
    self.bert_config.unpad = False
    self.bert_config.pad = False
    self.bert_config.fuse_qkv = True
    self.bert_config.fuse_scale = True
    self.bert_config.fuse_mask = True
    self.bert_config.fuse_dropout = False
    self.bert_config.apex_softmax = True
    self.bert_config.enable_stream = False
    self.bert_config.fused_dropout_add = False

    self.n_encoder_layers = 24
    self.encoder_layers = [BertLayer(self.bert_config) for x in range(self.n_encoder_layers)]
    
    # Disable things like dropout
    for layer in self.encoder_layers:
      layer.eval()

    self.tf_input_mask_path = self.data_path + 'input_tensors/input_mask.np'
    self.tf_input_mask = torch.from_numpy(np.load(self.tf_input_mask_path)).to(self.device)

    tf_tensors_path = self.data_path + 'intermediate_tensors/all_intermediate_tensors.pkl'
    if not gv.tf_tensors:
      with open(tf_tensors_path, mode='rb') as f:
        gv.tf_tensors = pickle.load(f)

    # Tensor keys needed for retrieval
    self.encoder_key_template = {
      'bert/encoder/layer_%/attention/self/from_tensor' : None,
      'bert/encoder/layer_%/attention/self/query_output' : None,
      'bert/encoder/layer_%/attention/self/key_output' : None,
      'bert/encoder/layer_%/attention/self/value_output' : None,
      'bert/encoder/layer_%/attention/self/attention_score_output' : None,
      'bert/encoder/layer_%/attention/self/attention_score_scaled_output' : None,
      'bert/encoder/layer_%/attention/self/attention_score_additive_output' : None,
      'bert/encoder/layer_%/attention/self/attention_probs_output' : None,
      'bert/encoder/layer_%/attention/self/context_layer' : None
    }

    # Check to see if this is the first setup to load the TF weights, otherwise just use the already loaded data
    if not gv.tf_weights:
      with open(self.data_path + 'tf_checkpoint.pkl', mode='rb') as f:
        gv.tf_weights = pickle.load(f)

    self.tf_encoder_tensors = []

    for layer_idx in range(self.n_encoder_layers):
      self.tf_encoder_tensors.append({key.replace('%', str(layer_idx)):torch.from_numpy(gv.tf_tensors[key.replace('%', str(layer_idx))]).to(self.device) for (key,value) in self.encoder_key_template.items()})

    # Weight keys needed for retrieval
    for layer_idx, encoder_layer in enumerate(self.encoder_layers):
      state_dict = encoder_layer.state_dict()
      pyt_tf_map = unit_test_utils.pyt_tf_mapping(state_dict, 'bert/encoder/layer_' + str(layer_idx) + '/')
      for key in pyt_tf_map:
        mapped_key = pyt_tf_map[key]
        if 'dense' in mapped_key or 'query' in mapped_key or 'key' in mapped_key or 'value' in mapped_key:
          state_dict[key] = torch.from_numpy(np.transpose(gv.tf_weights[mapped_key]))
        else:
          state_dict[key] = torch.from_numpy(gv.tf_weights[mapped_key])
      self.encoder_layers[layer_idx].load_state_dict(state_dict)

      self.encoder_layers[layer_idx] = encoder_layer.to(self.device)

    self.tolerance = 1e-5

  def test_encoders_initialized_properly(self):
    test_result = True

    for layer_idx, encoder_layer in enumerate(self.encoder_layers):
      state_dict = encoder_layer.state_dict()
      pyt_tf_map = unit_test_utils.pyt_tf_mapping(state_dict, 'bert/encoder/layer_' + str(layer_idx) + '/')
      for key in pyt_tf_map:
        mapped_key = pyt_tf_map[key]
        if 'dense' in mapped_key or 'query' in mapped_key or 'key' in mapped_key or 'value' in mapped_key:
          new_result = torch.equal(state_dict[key], torch.from_numpy(np.transpose(gv.tf_weights[mapped_key])).to(self.device))
          test_result = test_result and new_result
          if not new_result:
            print("Failure:", mapped_key)
        else:
          new_result = torch.equal(state_dict[key], torch.from_numpy(gv.tf_weights[mapped_key]).to(self.device))
          test_result = test_result and new_result
          if not new_result:
            print("Failure:", mapped_key)

    self.assertTrue(test_result)
  @parameterized.expand([(x,) for x in range(24)])
  def test_encoder_attention_query(self, layer_idx=0):
    pyt_result = self.encoder_layers[layer_idx].attention.self.query(self.tf_encoder_tensors[layer_idx]['bert/encoder/layer_' + str(layer_idx) + '/attention/self/from_tensor'])
    pyt_result = pyt_result.detach().cpu().numpy()
    
    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, self.tf_encoder_tensors[layer_idx]['bert/encoder/layer_' + str(layer_idx) + '/attention/self/query_output'].view(1,512,1024).detach().cpu().numpy())

    self.assertTrue(bins_absolute[-1] < self.tolerance)

  @parameterized.expand([(x,) for x in range(24)])
  def test_encoder_attention_key(self, layer_idx=0):
    pyt_result = self.encoder_layers[layer_idx].attention.self.key(self.tf_encoder_tensors[layer_idx]['bert/encoder/layer_' + str(layer_idx) + '/attention/self/from_tensor'])
    pyt_result = pyt_result.detach().cpu().numpy()

    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, self.tf_encoder_tensors[layer_idx]['bert/encoder/layer_' + str(layer_idx) + '/attention/self/key_output'].view(1,512,1024).detach().cpu().numpy())

    self.assertTrue(bins_absolute[-1] < self.tolerance)

  @parameterized.expand([(x,) for x in range(24)])
  def test_encoder_attention_value(self, layer_idx=0):
    pyt_result = self.encoder_layers[layer_idx].attention.self.value(self.tf_encoder_tensors[layer_idx]['bert/encoder/layer_' + str(layer_idx) + '/attention/self/from_tensor'])
    pyt_result = pyt_result.detach().cpu().numpy()

    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, self.tf_encoder_tensors[layer_idx]['bert/encoder/layer_' + str(layer_idx) + '/attention/self/value_output'].view(1,512,1024).detach().cpu().numpy())

    self.assertTrue(bins_absolute[-1] < self.tolerance)


if __name__ == '__main__':
  unittest.main(verbosity=2)

