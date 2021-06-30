# NVIDIA

import math
import numpy as np
import pickle
import unittest

from parameterized import parameterized, parameterized_class

import test_data_path
import unit_test_utils
import global_vars as gv

from modeling import BertConfig, BertForPretraining

import torch
from torch import nn

class TestBert(unittest.TestCase):
  @classmethod
  def setUpClass(self):
    self.device = 'cuda:0'
    self.data_path = test_data_path.get_path() + '/gold/batch_1/'
    self.config_path = '/workspace/phase1/bert_config.json'
    self.bert_config = BertConfig(vocab_size_or_config_json_file=self.config_path)
    self.tf_checkpoint_path = '/workspace/phase1/model.ckpt-28252'

    # Additional config options
    self.bert_config.fused_mha = False
    self.bert_config.fused_gelu_bias = True
    self.bert_config.dense_seq_output = True # True
    self.bert_config.unpad = False
    self.bert_config.pad = False
    self.bert_config.fuse_qkv = True
    self.bert_config.fuse_scale = True
    self.bert_config.fuse_mask = True
    self.bert_config.fuse_dropout = False
    self.bert_config.apex_softmax = True
    self.bert_config.enable_stream = False
    self.bert_config.fused_dropout_add = False

    if not gv.pyt_model:
      gv.pyt_model = BertForPretraining.from_pretrained(self.tf_checkpoint_path, from_tf=True, config=self.bert_config)
    self.bert = gv.pyt_model

    #self.bert = BertForPretraining.from_pretrained(self.tf_checkpoint_path, from_tf=True, config=self.bert_config)
    
    # Disable things like dropout
    self.bert.eval()

    self.tf_input_mask_path = self.data_path + 'input_tensors/input_mask.np'
    self.tf_input_mask = torch.from_numpy(np.load(self.tf_input_mask_path)).to(self.device)

    tf_tensors_path = self.data_path + 'intermediate_tensors/all_intermediate_tensors.pkl'
    if not gv.tf_tensors:
      with open(tf_tensors_path, mode='rb') as f:
        gv.tf_tensors = pickle.load(f)

    # Tensor keys needed for retrieval
    self.tf_input_ids_path = self.data_path + 'input_tensors/input_ids.np'
    self.tf_input_ids = torch.from_numpy(np.load(self.tf_input_ids_path)).to(self.device)

    self.tf_token_type_ids_path = self.data_path + 'input_tensors/segment_ids.np'
    self.tf_token_type_ids = torch.from_numpy(np.load(self.tf_token_type_ids_path)).to(self.device)

    self.tf_masked_lm_positions_path = self.data_path + 'input_tensors/masked_lm_positions.np'
    self.tf_masked_lm_positions = torch.from_numpy(np.load(self.tf_masked_lm_positions_path)).to(self.device)

    self.tf_masked_lm_ids_path = self.data_path + 'input_tensors/masked_lm_ids.np'
    self.tf_masked_lm_ids = torch.from_numpy(np.load(self.tf_masked_lm_ids_path)).to(self.device)

    self.tf_next_sentence_label_path = self.data_path + 'input_tensors/next_sentence_labels.np'
    self.tf_next_sentence_label = torch.from_numpy(np.load(self.tf_next_sentence_label_path)).to(self.device)

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

    self.tf_encoder_tensors = []
    self.n_encoder_layers = 24
    for layer_idx in range(self.n_encoder_layers):
      self.tf_encoder_tensors.append({key.replace('%', str(layer_idx)):torch.from_numpy(gv.tf_tensors[key.replace('%', str(layer_idx))]).to(self.device) for (key,value) in self.encoder_key_template.items()})


    # Check to see if this is the first setup to load the TF weights, otherwise just use the already loaded data
    if not gv.tf_weights:
      with open(self.data_path + 'tf_checkpoint.pkl', mode='rb') as f:
        gv.tf_weights = pickle.load(f)

    # Weight keys needed for retrieval
    state_dict = self.bert.state_dict()
    pyt_tf_map = unit_test_utils.pyt_tf_mapping(state_dict)
    for key in pyt_tf_map:
      mapped_key = pyt_tf_map[key]
      if 'dense' in mapped_key or 'query' in mapped_key or 'key' in mapped_key or 'value' in mapped_key:
        state_dict[key] = torch.from_numpy(np.transpose(gv.tf_weights[mapped_key]))
      else:
        state_dict[key] = torch.from_numpy(gv.tf_weights[mapped_key])

    self.bert = self.bert.to(self.device)

    self.tolerance = 5e-5


  def test_bert_word_embeddings(self):
    pyt_result = self.bert.bert.embeddings.word_embeddings(self.tf_input_ids.long())
    pyt_result = pyt_result.detach().cpu().numpy()

    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, self.tf_word_embeddings_output)
    self.assertTrue(bins_absolute[-1] < self.tolerance)


  def test_bert_position_embeddings(self):
    pyt_result = self.bert.bert.embeddings.position_embeddings(self.bert.bert.embeddings.get_position_ids(self.tf_input_ids.long()))
    pyt_result = pyt_result.detach().cpu().numpy()

    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, self.tf_position_embeddings_output)
    self.assertTrue(bins_absolute[-1] < self.tolerance)


  def test_bert_token_type_embeddings(self):
    pyt_result = self.bert.bert.embeddings.token_type_embeddings(self.tf_token_type_ids.long())
    pyt_result = pyt_result.detach().cpu().numpy()

    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, self.tf_token_type_embeddings_output)
    self.assertTrue(bins_absolute[-1] < self.tolerance)


  def test_bert_embeddings_summation(self):
    pyt_result = self.bert.bert.embeddings.word_embeddings(self.tf_input_ids.long()) \
               + self.bert.bert.embeddings.position_embeddings(self.bert.bert.embeddings.get_position_ids(self.tf_input_ids.long())) \
               + self.bert.bert.embeddings.token_type_embeddings(self.tf_token_type_ids.long())
    pyt_result = pyt_result.detach().cpu().numpy()

    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, self.tf_embedding_summation_output)
    self.assertTrue(bins_absolute[-1] < self.tolerance)


  def test_bert_embedding(self):
    pyt_result = self.bert.bert.embeddings(self.tf_input_ids.long(), self.tf_token_type_ids.long())
    pyt_result = pyt_result.detach().cpu().numpy()

    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, self.tf_embedding_output)
    self.assertTrue(bins_absolute[-1] < self.tolerance)


  @parameterized.expand([(x,) for x in range(24)])
  def test_bert_encoder_attention_query(self, layer_idx=0):
    pyt_result = self.bert.bert.encoder.layer[layer_idx].attention.self.query(self.tf_encoder_tensors[layer_idx]['bert/encoder/layer_' + str(layer_idx) + '/attention/self/from_tensor'])
    pyt_result = pyt_result.detach().cpu().numpy()

    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, self.tf_encoder_tensors[layer_idx]['bert/encoder/layer_' + str(layer_idx) + '/attention/self/query_output'].view(1,512,1024).detach().cpu().numpy())
    
    test_pass = bins_absolute[-1] < self.tolerance
    
    if not test_pass:
      print('max abs diff:', bins_absolute[-1])

    self.assertTrue(test_pass)


  @parameterized.expand([(x,) for x in range(24)])
  def test_bert_encoder_attention_key(self, layer_idx=0):
    pyt_result = self.bert.bert.encoder.layer[layer_idx].attention.self.key(self.tf_encoder_tensors[layer_idx]['bert/encoder/layer_' + str(layer_idx) + '/attention/self/from_tensor'])
    pyt_result = pyt_result.detach().cpu().numpy()

    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, self.tf_encoder_tensors[layer_idx]['bert/encoder/layer_' + str(layer_idx) + '/attention/self/key_output'].view(1,512,1024).detach().cpu().numpy())

    test_pass = bins_absolute[-1] < self.tolerance

    if not test_pass:
      print('max abs diff:', bins_absolute[-1])

    self.assertTrue(test_pass)


  @parameterized.expand([(x,) for x in range(24)])
  def test_bert_encoder_attention_value(self, layer_idx=0):
    pyt_result = self.bert.bert.encoder.layer[layer_idx].attention.self.value(self.tf_encoder_tensors[layer_idx]['bert/encoder/layer_' + str(layer_idx) + '/attention/self/from_tensor'])
    pyt_result = pyt_result.detach().cpu().numpy()

    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, self.tf_encoder_tensors[layer_idx]['bert/encoder/layer_' + str(layer_idx) + '/attention/self/value_output'].view(1,512,1024).detach().cpu().numpy())

    test_pass = bins_absolute[-1] < self.tolerance

    if not test_pass:
      print('max abs diff:', bins_absolute[-1])

    self.assertTrue(test_pass)


  @parameterized.expand([(x,) for x in range(24)])
  def test_bert_encoder_self_attention_output(self, layer_idx=0):
    model_updated_input_mask = (1.0 - self.tf_input_mask) * -10000.0
    pyt_result = self.bert.bert.encoder.layer[layer_idx].attention.self(
        self.tf_encoder_tensors[layer_idx]['bert/encoder/layer_' + str(layer_idx) + '/attention/self/from_tensor'],
        model_updated_input_mask)
    pyt_result = pyt_result.detach().cpu().numpy()

    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, self.tf_encoder_tensors[layer_idx]['bert/encoder/layer_' + str(layer_idx) + '/attention/self/context_layer'].view(1,512,1024).detach().cpu().numpy())

    test_pass = bins_absolute[-1] < self.tolerance

    if not test_pass:
      print('max abs diff:', bins_absolute[-1])

    self.assertTrue(test_pass)


  @parameterized.expand([(x,) for x in range(24)])
  def test_bert_encoder_self_attention_score(self, layer_idx=0):
    q = self.bert.bert.encoder.layer[layer_idx].attention.self.query(
        self.tf_encoder_tensors[layer_idx]['bert/encoder/layer_' + str(layer_idx) + '/attention/self/from_tensor'])
    k = self.bert.bert.encoder.layer[layer_idx].attention.self.key(
        self.tf_encoder_tensors[layer_idx]['bert/encoder/layer_' + str(layer_idx) + '/attention/self/from_tensor'])
    v = self.bert.bert.encoder.layer[layer_idx].attention.self.value(
        self.tf_encoder_tensors[layer_idx]['bert/encoder/layer_' + str(layer_idx) + '/attention/self/from_tensor'])

    q_t = self.bert.bert.encoder.layer[layer_idx].attention.self.transpose_for_scores(q)
    k_t = self.bert.bert.encoder.layer[layer_idx].attention.self.transpose_key_for_scores(k)
    v_t = self.bert.bert.encoder.layer[layer_idx].attention.self.transpose_for_scores(v)

    pyt_result = torch.matmul(q_t, k_t).detach().cpu().numpy()
  
    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, self.tf_encoder_tensors[layer_idx]['bert/encoder/layer_' + str(layer_idx) + '/attention/self/attention_score_output'].detach().cpu().numpy())

    test_pass = bins_absolute[-1] < 2e-4

    if not test_pass:
      print('max abs diff:', bins_absolute[-1])

    self.assertTrue(test_pass)
  

  @parameterized.expand([(x,) for x in range(24)])
  def test_bert_encoder_self_attention_score_normalized(self, layer_idx=0):
    q = self.bert.bert.encoder.layer[layer_idx].attention.self.query(
        self.tf_encoder_tensors[layer_idx]['bert/encoder/layer_' + str(layer_idx) + '/attention/self/from_tensor'])
    k = self.bert.bert.encoder.layer[layer_idx].attention.self.key(
        self.tf_encoder_tensors[layer_idx]['bert/encoder/layer_' + str(layer_idx) + '/attention/self/from_tensor'])
    v = self.bert.bert.encoder.layer[layer_idx].attention.self.value(
        self.tf_encoder_tensors[layer_idx]['bert/encoder/layer_' + str(layer_idx) + '/attention/self/from_tensor'])

    q_t = self.bert.bert.encoder.layer[layer_idx].attention.self.transpose_for_scores(q)
    k_t = self.bert.bert.encoder.layer[layer_idx].attention.self.transpose_key_for_scores(k)
    v_t = self.bert.bert.encoder.layer[layer_idx].attention.self.transpose_for_scores(v)

    pyt_result = torch.matmul(q_t, k_t)
    pyt_result = pyt_result / math.sqrt(self.bert.bert.encoder.layer[layer_idx].attention.self.attention_head_size)
    pyt_result = pyt_result - 10000.0 * (1.0 - self.tf_input_mask.float().unsqueeze(1).unsqueeze(2))
    pyt_result = pyt_result.detach().cpu().numpy()

    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, self.tf_encoder_tensors[layer_idx]['bert/encoder/layer_' + str(layer_idx) + '/attention/self/attention_score_additive_output'].detach().cpu().numpy())

    test_pass = bins_absolute[-1] < 1e-3

    if not test_pass:
      print('max abs diff:', bins_absolute[-1])

    self.assertTrue(test_pass)


  @parameterized.expand([(x,) for x in range(24)]) 
  def test_bert_encoder_self_attention_probs(self, layer_idx=0):
    q = self.bert.bert.encoder.layer[layer_idx].attention.self.query(
        self.tf_encoder_tensors[layer_idx]['bert/encoder/layer_' + str(layer_idx) + '/attention/self/from_tensor'])
    k = self.bert.bert.encoder.layer[layer_idx].attention.self.key(
        self.tf_encoder_tensors[layer_idx]['bert/encoder/layer_' + str(layer_idx) + '/attention/self/from_tensor'])
    v = self.bert.bert.encoder.layer[layer_idx].attention.self.value(
        self.tf_encoder_tensors[layer_idx]['bert/encoder/layer_' + str(layer_idx) + '/attention/self/from_tensor'])

    q_t = self.bert.bert.encoder.layer[layer_idx].attention.self.transpose_for_scores(q)
    k_t = self.bert.bert.encoder.layer[layer_idx].attention.self.transpose_key_for_scores(k)
    v_t = self.bert.bert.encoder.layer[layer_idx].attention.self.transpose_for_scores(v)

    pyt_result = torch.matmul(q_t, k_t)
    pyt_result = pyt_result / math.sqrt(self.bert.bert.encoder.layer[layer_idx].attention.self.attention_head_size)
    pyt_result = pyt_result - 10000.0 * (1.0 - self.tf_input_mask.float().unsqueeze(1).unsqueeze(2))
    pyt_result = self.bert.bert.encoder.layer[layer_idx].attention.self.softmax(pyt_result)
    pyt_result = pyt_result.detach().cpu().numpy()

    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, self.tf_encoder_tensors[layer_idx]['bert/encoder/layer_' + str(layer_idx) + '/attention/self/attention_probs_output'].detach().cpu().numpy())

    test_pass = bins_absolute[-1] < self.tolerance

    if not test_pass:
      print('max abs diff:', bins_absolute[-1])

    self.assertTrue(test_pass)


  @parameterized.expand([(x,) for x in range(24)])
  def test_bert_encoder_layers(self, layer_idx=0):
    seq_len = 512
    batch_size = 1
    model_updated_input_mask = (1.0 - self.tf_input_mask) * -10000.0
    pyt_result = self.bert.bert.encoder.layer[layer_idx](self.tf_encoder_tensors[layer_idx]['bert/encoder/layer_' + str(layer_idx) + '/attention/self/from_tensor'], model_updated_input_mask, seq_len, batch_size)
    pyt_result = pyt_result.detach().cpu().numpy()

    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, torch.from_numpy(gv.tf_tensors['bert/encoder/layer_' + str(layer_idx) + '/output/layer_output']).view(1,512,1024).numpy())

    test_pass = bins_absolute[-1] < self.tolerance

    if not test_pass:
      print('bert.encoder.layer.0 max abs diff:', bins_absolute[-1])

    self.assertTrue(test_pass)


  def test_bert_pooler_isolated(self):
    pyt_result = self.bert.bert.pooler(torch.from_numpy(gv.tf_tensors['bert/encoder/layer_23/output/layer_output']).to(self.device))
    pyt_result = pyt_result.detach().cpu().numpy()

    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, gv.tf_tensors['bert/pooler/pooler_output']) 

    test_pass = bins_absolute[-1] < self.tolerance

    if not test_pass:
      print('bert.pooler max abs diff:', bins_absolute[-1])

    self.assertTrue(test_pass)

  def test_bert_total_loss(self):
    input_ids = self.tf_input_ids.long().view(-1)
    segment_ids = self.tf_token_type_ids.long().view(-1)
    input_mask = self.tf_input_mask.long().view(-1)
    masked_lm_positions = self.tf_masked_lm_positions.long().view(-1)
    masked_lm_ids = self.tf_masked_lm_ids.long().view(-1)
    next_sentence_label = self.tf_next_sentence_label.long().view(-1)
    
    masked_lm_labels = torch.zeros(input_ids.shape, dtype=torch.long).to(self.device)
    index = masked_lm_positions[masked_lm_positions != 0].shape[-1]
    masked_token_count = torch.count_nonzero(masked_lm_positions)
    if masked_token_count != 0:
      index = masked_token_count
    masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

    input_ids = input_ids.view(1, -1)
    segment_ids = segment_ids.view(1, -1)
    input_mask = input_mask.view(1, -1)
    masked_lm_positions = masked_lm_positions.view(1, -1)
    masked_lm_ids = masked_lm_ids.view(1, -1)
    next_sentence_label = next_sentence_label.view(1, -1)

    masked_lm_labels = masked_lm_labels.reshape(1, 512)

    loss, mlm_acc, num_masked = self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, masked_lm_labels=masked_lm_labels, next_sentence_label=next_sentence_label)

    pyt_result = loss
    pyt_result = pyt_result.detach().cpu().numpy()
    
    tf_result = gv.tf_tensors['cls/predictions/mlm_loss'] + gv.tf_tensors['cls/seq_relationship/nsp_loss']

    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, tf_result)

    test_pass = bins_absolute[-1] < self.tolerance

    if not test_pass:
      print('bert total loss max abs diff:', bins_absolute[-1])
      print('pyt:', pyt_result)
      print('tf:', tf_result)

    self.assertTrue(test_pass)


  def test_bert_mlm_loss(self):
    sequence_output = torch.from_numpy(gv.tf_tensors['cls/predictions/mlm_input_tensor']).to(self.device)
    pooled_output = torch.from_numpy(gv.tf_tensors['bert/pooler/pooler_output']).to(self.device)
    input_ids = self.tf_input_ids.long().view(-1)
    masked_lm_positions = self.tf_masked_lm_positions.long().view(-1)
    masked_lm_ids = self.tf_masked_lm_ids.long().view(-1)

    masked_lm_labels = torch.zeros(input_ids.shape, dtype=torch.long).to(self.device)
    index = masked_lm_positions[masked_lm_positions != 0].shape[-1]
    masked_token_count = torch.count_nonzero(masked_lm_positions)
    if masked_token_count != 0:
      index = masked_token_count
    masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

    masked_lm_positions = masked_lm_positions.view(1, -1)
    masked_lm_ids = masked_lm_ids.view(1, -1)

    prediction_scores, seq_relationship_score = self.bert.cls(sequence_output, pooled_output, masked_lm_labels)

    masked_lm_labels_flat = masked_lm_labels.view(-1)
    
    if self.bert_config.dense_seq_output:
        masked_lm_labels_flat = masked_lm_labels_flat[masked_lm_labels_flat != 0]

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)

    pyt_result = loss_fct(prediction_scores.view(-1, self.bert_config.vocab_size), masked_lm_labels_flat.view(-1))
    pyt_result = pyt_result.detach().cpu().numpy()

    tf_result = gv.tf_tensors['cls/predictions/mlm_loss']
    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, tf_result)

    test_pass = bins_absolute[-1] < self.tolerance

    if not test_pass:
      print('bert mlm loss max abs diff:', bins_absolute[-1])
      print('pyt:', pyt_result)
      print('tf:', tf_result)

    self.assertTrue(test_pass)


  def test_bert_model_sequence_output(self):
    input_ids = self.tf_input_ids.long()
    segment_ids = self.tf_token_type_ids.long()
    input_mask = self.tf_input_mask.long()
    masked_lm_labels = self.tf_masked_lm_ids.long()
    next_sentence_label = self.tf_next_sentence_label.long()

    sequence_output, pooled_output = self.bert.bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)

    pyt_result = sequence_output
    pyt_result = pyt_result.detach().cpu().numpy()

    tf_result = gv.tf_tensors['bert/encoder/layer_23/output/layer_output']  # TODO: update this

    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, tf_result)

    test_pass = bins_absolute[-1] < 9.5e-5  # A100 exception

    if not test_pass:
      print('bert sequence output max abs diff:', bins_absolute[-1], '@', max_idx)
      print('pyt:', pyt_result)
      print('tf:', tf_result)

    self.assertTrue(test_pass)


  def test_bert_mlm_compare_tf_tensors(self):
    sequence_output_1 = gv.tf_tensors['bert/encoder/layer_23/output/layer_output']
    sequence_output_2 = gv.tf_tensors['cls/predictions/mlm_input_tensor']

    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(sequence_output_1, sequence_output_2)

    test_pass = bins_absolute[-1] < self.tolerance

    if not test_pass:
      print('bert tf tensors max abs diff:', bins_absolute[-1])
      print('tf1:', sequence_output_1)
      print('tf2:', sequence_output_2)

    self.assertTrue(test_pass)


  def test_bert_mlm_prediction_scores(self):
    sequence_output = torch.from_numpy(gv.tf_tensors['cls/predictions/mlm_input_tensor']).to(self.device)
    pooled_output = torch.from_numpy(gv.tf_tensors['bert/pooler/pooler_output']).to(self.device)
    input_ids = self.tf_input_ids.long().view(-1)
    masked_lm_positions = self.tf_masked_lm_positions.long().view(-1)
    masked_lm_ids = self.tf_masked_lm_ids.long().view(-1)

    masked_lm_labels = torch.zeros(input_ids.shape, dtype=torch.long).to(self.device)
    index = masked_lm_positions[masked_lm_positions != 0].shape[-1]
    masked_token_count = torch.count_nonzero(masked_lm_positions)
    if masked_token_count != 0:
      index = masked_token_count
    masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

    masked_lm_positions = masked_lm_positions.view(1, -1)
    masked_lm_labels = masked_lm_labels.view(1, -1)
    masked_lm_labels = masked_lm_labels.reshape(1,-1)

    prediction_scores, seq_relationship_score = self.bert.cls(sequence_output, pooled_output, masked_lm_labels)

    pyt_result = prediction_scores
    pyt_result = pyt_result.detach().cpu().numpy()

    tf_result = nn.functional.pad(torch.from_numpy(gv.tf_tensors['cls/predictions/mlm_logits_bias']), (0,6), "constant", 0)
    tf_result = tf_result[masked_lm_positions.view(-1).nonzero().squeeze(1),:]
    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, tf_result.detach().cpu().numpy())

    test_pass = bins_absolute[-1] < self.tolerance

    if not test_pass:
      print('bert mlm prediction scores max abs diff:', bins_absolute[-1])
      print('pyt:', pyt_result)
      print('tf:', tf_result)
      print('prediction_scores.shape:', prediction_scores.shape)
      print('cls/predictions/mlm_logits_matmul.shape:', tf_result.shape)

    self.assertTrue(test_pass)


  def test_bert_mlm_transform(self):
    sequence_output = torch.from_numpy(gv.tf_tensors['cls/predictions/mlm_input_tensor_gather']).to(self.device)

    pyt_result = self.bert.cls.predictions.transform.dense(sequence_output)
    pyt_result = self.bert.cls.predictions.transform.transform_act_fn(pyt_result)
    pyt_result = pyt_result.detach().cpu().numpy()

    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, gv.tf_tensors['cls/predictions/transform/mlm_input_tensor_transformed'])

    test_pass = bins_absolute[-1] < self.tolerance

    if not test_pass:
      print('bert mlm transform max abs diff:', bins_absolute[-1])
      print('pyt:', pyt_result)
      print('tf:', gv.tf_tensors['cls/predictions/transform/mlm_input_tensor_transformed'])

    self.assertTrue(test_pass)


  def test_bert_mlm_layernorm(self):
    sequence_output = torch.from_numpy(gv.tf_tensors['cls/predictions/mlm_input_tensor_gather']).to(self.device)

    pyt_result = self.bert.cls.predictions.transform(sequence_output)
    pyt_result = pyt_result.detach().cpu().numpy()

    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, gv.tf_tensors['cls/predictions/transform/mlm_input_tensor_layernorm'])

    test_pass = bins_absolute[-1] < self.tolerance

    if not test_pass:
      print('bert mlm transform layernorm max abs diff:', bins_absolute[-1])
      print('pyt:', pyt_result)
      print('tf:', gv.tf_tensors['cls/predictions/transform/mlm_input_tensor_layernorm'])

    self.assertTrue(test_pass)


  def test_bert_mlm_decoder_matmul(self):
    sequence_output = torch.from_numpy(gv.tf_tensors['cls/predictions/mlm_input_tensor_gather']).to(self.device)

    pyt_result = self.bert.cls.predictions.transform(sequence_output)
    pyt_result = self.bert.cls.predictions.decoder(pyt_result)
    pyt_result = pyt_result.detach().cpu().numpy()

    tf_result = nn.functional.pad(torch.from_numpy(gv.tf_tensors['cls/predictions/mlm_logits_matmul']), (0,6), "constant", 0)

    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, tf_result.numpy())

    test_pass = bins_absolute[-1] < self.tolerance

    if not test_pass:
      print('bert mlm decoder matmul max abs diff:', bins_absolute[-1])
      print('pyt:', pyt_result)
      print('tf:', gv.tf_tensors['cls/predictions/mlm_logits_matmul'])

    self.assertTrue(test_pass)


  def test_bert_mlm_decoder_bias(self):
    sequence_output = torch.from_numpy(gv.tf_tensors['cls/predictions/mlm_input_tensor_gather']).to(self.device)

    pyt_result = self.bert.cls.predictions(sequence_output)
    pyt_result = pyt_result.detach().cpu().numpy()

    tf_result = nn.functional.pad(torch.from_numpy(gv.tf_tensors['cls/predictions/mlm_logits_bias']), (0,6), "constant", 0)

    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, tf_result.numpy())

    test_pass = bins_absolute[-1] < self.tolerance

    if not test_pass:
      print('bert mlm decoder bias max abs diff:', bins_absolute[-1])
      print('pyt:', pyt_result)
      print('tf:', gv.tf_tensors['cls/predictions/mlm_logits_bias'])

    self.assertTrue(test_pass)


  def test_bert_nsp_loss(self):
    sequence_output = torch.from_numpy(gv.tf_tensors['bert/encoder/layer_23/output/layer_output']).to(self.device)
    pooled_output = torch.from_numpy(gv.tf_tensors['bert/pooler/pooler_output']).to(self.device)
    masked_lm_positions = self.tf_masked_lm_positions.long()
    masked_lm_labels = self.tf_masked_lm_ids.long()
    next_sentence_label = self.tf_next_sentence_label.long()

    prediction_scores, seq_relationship_score = self.bert.cls(sequence_output, pooled_output, masked_lm_positions)

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)

    pyt_result = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
    pyt_result = pyt_result.detach().cpu().numpy()

    counts, bins_absolute, bins_relative, max_idx = unit_test_utils.max_abs_diff_binning(pyt_result, gv.tf_tensors['cls/seq_relationship/nsp_loss'])

    test_pass = bins_absolute[-1] < self.tolerance

    if not test_pass:
      print('bert nsp loss max abs diff:', bins_absolute[-1])
      print('pyt:', pyt_result)
      print('tf:', gv.tf_tensors['cls/seq_relationship/nsp_loss'])

    self.assertTrue(test_pass)


  def SKIP_test_bert_initialized_properly(self):
    test_result = True

    state_dict = self.bert.state_dict()
    pyt_tf_map = unit_test_utils.pyt_tf_mapping(state_dict)
    for idx, key in enumerate(pyt_tf_map):
      mapped_key = pyt_tf_map[key]
      if 'dense' in mapped_key or 'query' in mapped_key or 'key' in mapped_key or 'value' in mapped_key:
        new_result = torch.equal(state_dict[key], torch.from_numpy(np.transpose(gv.tf_weights[mapped_key])).to(self.device))
        test_result = test_result and new_result
        if not new_result:
          print("Failure:", state_dict.keys()[idx], '-->', mapped_key)
      else:
        new_result = torch.equal(state_dict[key], torch.from_numpy(gv.tf_weights[mapped_key]).to(self.device))
        test_result = test_result and new_result
        if not new_result:
          print("Failure:", state_dict.keys()[idx], '-->', mapped_key)

    self.assertTrue(test_result)
  

if __name__ == '__main__':
  unittest.main(verbosity=2)

