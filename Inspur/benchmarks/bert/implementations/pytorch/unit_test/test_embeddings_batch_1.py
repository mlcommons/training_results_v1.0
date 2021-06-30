# NVIDIA

import numpy as np
import pickle
import unittest

import test_data_path
import unit_test_utils
import global_vars as gv

import model.layers.embeddings
from modeling import BertConfig

import torch
from torch import nn

class TestEmbeddings(unittest.TestCase):
  def setUp(self):
    self.device = 'cuda:0'
    self.data_path = test_data_path.get_path() + '/gold/batch_1/'
    self.config_path = '/workspace/phase1/bert_config.json'
    self.bert_config = BertConfig(vocab_size_or_config_json_file=self.config_path)
    self.bert_config.fused_dropout_add = False
    self.embedding_layer = model.layers.embeddings.BertEmbeddings(self.bert_config)
    
    # Disable things like dropout
    self.embedding_layer.eval()

    self.tf_input_ids_path = self.data_path + 'input_tensors/input_ids.np'
    self.tf_input_ids = torch.from_numpy(np.load(self.tf_input_ids_path)).to(self.device)

    self.tf_token_type_ids_path = self.data_path + 'input_tensors/segment_ids.np'
    self.tf_token_type_ids = torch.from_numpy(np.load(self.tf_token_type_ids_path)).to(self.device)

    self.tf_word_embeddings_output_path = self.data_path + 'intermediate_tensors/bert_embeddings_word_embedding_output.np'
    self.tf_word_embeddings_output = np.load(self.tf_word_embeddings_output_path)

    self.tf_position_embeddings_output_path = self.data_path + 'intermediate_tensors/bert_embeddings_position_embeddings.np'
    self.tf_position_embeddings_output = np.load(self.tf_position_embeddings_output_path)

    self.tf_token_type_embeddings_output_path = self.data_path + 'intermediate_tensors/bert_embeddings_token_type_embeddings_output.np'
    self.tf_token_type_embeddings_output = np.load(self.tf_token_type_embeddings_output_path)

    self.tf_embedding_summation_output_path = self.data_path + 'intermediate_tensors/bert_embeddings_embedding_summation_output.np'
    self.tf_embedding_summation_output = np.load(self.tf_embedding_summation_output_path)

    self.tf_embedding_output_path = self.data_path + 'intermediate_tensors/bert_embeddings_output.np'
    self.tf_embedding_output = np.load(self.tf_embedding_output_path)

    # Weight keys needed for retrieval
    self.word_embeddings_key = 'bert/embeddings/word_embeddings'
    self.position_embeddings_key = 'bert/embeddings/position_embeddings'
    self.token_type_embeddings_key = 'bert/embeddings/token_type_embeddings'
    self.layernorm_beta_key = 'bert/embeddings/LayerNorm/beta'
    self.layernorm_gamma_key = 'bert/embeddings/LayerNorm/gamma'

    # Check to see if this is the first setup to load the TF weights, otherwise just use the already loaded data
    if not gv.tf_weights:
      with open(self.data_path + 'tf_checkpoint.pkl', mode='rb') as f:
        gv.tf_weights = pickle.load(f)
    with torch.no_grad():
      self.tf_word_embeddings_weights = nn.Parameter(torch.from_numpy(gv.tf_weights[self.word_embeddings_key]))
      self.tf_position_embeddings_weights = nn.Parameter(torch.from_numpy(gv.tf_weights[self.position_embeddings_key]))
      self.tf_token_type_embeddings_weights = nn.Parameter(torch.from_numpy(gv.tf_weights[self.token_type_embeddings_key]))
      self.embedding_layer.LayerNorm.bias = nn.Parameter(torch.from_numpy(gv.tf_weights[self.layernorm_beta_key]))
      self.embedding_layer.LayerNorm.weight = nn.Parameter(torch.from_numpy(gv.tf_weights[self.layernorm_gamma_key]))

    self.embedding_layer.word_embeddings.weight = self.tf_word_embeddings_weights
    self.embedding_layer.position_embeddings.weight = self.tf_position_embeddings_weights
    self.embedding_layer.token_type_embeddings.weight = self.tf_token_type_embeddings_weights

    self.embedding_layer = self.embedding_layer.to(self.device)

    self.tolerance = 1e-5

  def test_word_embeddings(self):
    pyt_result = self.embedding_layer.word_embeddings(self.tf_input_ids.long())
    pyt_result = pyt_result.detach().cpu().numpy()

    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, self.tf_word_embeddings_output)
    self.assertTrue(bins_absolute[-1] < self.tolerance)
 
  def test_position_embeddings(self):
    pyt_result = self.embedding_layer.position_embeddings(self.embedding_layer.get_position_ids(self.tf_input_ids.long()))
    pyt_result = pyt_result.detach().cpu().numpy()

    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, self.tf_position_embeddings_output)
    self.assertTrue(bins_absolute[-1] < self.tolerance)

  def test_token_type_embeddings(self):
    pyt_result = self.embedding_layer.token_type_embeddings(self.tf_token_type_ids.long())
    pyt_result = pyt_result.detach().cpu().numpy()

    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, self.tf_token_type_embeddings_output)
    self.assertTrue(bins_absolute[-1] < self.tolerance)
  
  def test_embeddings_summation(self):
    pyt_result = self.embedding_layer.word_embeddings(self.tf_input_ids.long()) \
               + self.embedding_layer.position_embeddings(self.embedding_layer.get_position_ids(self.tf_input_ids.long())) \
               + self.embedding_layer.token_type_embeddings(self.tf_token_type_ids.long())
    pyt_result = pyt_result.detach().cpu().numpy()

    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, self.tf_embedding_summation_output)
    self.assertTrue(bins_absolute[-1] < self.tolerance)

  def test_embedding(self):
    pyt_result = self.embedding_layer(self.tf_input_ids.long(), self.tf_token_type_ids.long())
    pyt_result = pyt_result.detach().cpu().numpy()

    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, self.tf_embedding_output)
    self.assertTrue(bins_absolute[-1] < self.tolerance)
    

if __name__ == '__main__':
  unittest.main(verbosity=2)

