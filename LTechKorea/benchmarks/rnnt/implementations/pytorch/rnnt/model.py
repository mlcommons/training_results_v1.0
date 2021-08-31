# Copyright (c) 2019, Myrtle Software Limited. All rights reserved.
# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlperf import logging
from apex.mlp import MLP

from common.rnn import rnn
from apex.contrib.transducer import TransducerJoint

torch._C._jit_set_nvfuser_enabled(True)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_bailout_depth(20)

class StackTime(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = int(factor)

    def stack(self, x):
        # x is [T, B, H]
        x = x.transpose(0, 1)
        T = x.size(1)
        padded = torch.nn.functional.pad(x, (0, 0, 0, (self.factor - (T % self.factor)) % self.factor))
        B, T, H = padded.size()
        x = padded.reshape(B, T // self.factor, -1)
        x = x.transpose(0, 1)
        return x

        
    def forward(self, x, x_lens):
        # T, B, U
        if type(x) is not list:
            x = self.stack(x)
            x_lens = (x_lens.int() + self.factor - 1) // self.factor
            return x, x_lens
        else:
            # seq splitting path
            if len(x) != 2:
                raise NotImplementedError("Only number of seq segments equal to 2 is supported")
            # For simplicty, we assume except or the last segment, all seq segments should be 
            # multiple of self.factor. Therefore, we can ensure each segment can be done independently.
            assert x[0].size(1) % self.factor == 0, "The length of the 1st seq segment should be multiple of stack factor"
            y0 = self.stack(x[0])
            y1 = self.stack(x[1])
            x_lens = (x_lens.int() + self.factor - 1) // self.factor
            return [y0, y1], x_lens

@torch.jit.script
def jit_relu_dropout(x, prob) :
    # type: (Tensor, float) -> Tensor
    x = torch.nn.functional.relu(x)
    x = torch.nn.functional.dropout(x, p=prob, training=True)
    return x

class FusedReluDropout(nn.Module):
    def __init__(self, prob):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        if self.training:
            return jit_relu_dropout(x, self.prob)
        else:
            return torch.nn.functional.relu(x)


class RNNT(nn.Module):
    """A Recurrent Neural Network Transducer (RNN-T).

    Args:
        in_features: Number of input features per step per batch.
        vocab_size: Number of output symbols (inc blank).
        forget_gate_bias: Total initialized value of the bias used in the
            forget gate. Set to None to use PyTorch's default initialisation.
            (See: http://proceedings.mlr.press/v37/jozefowicz15.pdf)
        batch_norm: Use batch normalization in encoder and prediction network
            if true.
        encoder_n_hidden: Internal hidden unit size of the encoder.
        encoder_rnn_layers: Encoder number of layers.
        pred_n_hidden:  Internal hidden unit size of the prediction network.
        pred_rnn_layers: Prediction network number of layers.
        joint_n_hidden: Internal hidden unit size of the joint network.
    """
    def __init__(self, n_classes, in_feats, enc_n_hid,
                 enc_pre_rnn_layers, enc_post_rnn_layers, enc_stack_time_factor,
                 enc_dropout, pred_dropout, joint_dropout,
                 pred_n_hid, pred_rnn_layers, joint_n_hid,
                 forget_gate_bias, decoupled_rnns,
                 hidden_hidden_bias_scale=0.0, weights_init_scale=1.0,
                 enc_lr_factor=1.0, pred_lr_factor=1.0, joint_lr_factor=1.0,
                 fuse_relu_dropout=False, apex_transducer_joint=None,
                 min_lstm_bs=8, apex_mlp=False):
        super(RNNT, self).__init__()

        self.enc_lr_factor = enc_lr_factor
        self.pred_lr_factor = pred_lr_factor
        self.joint_lr_factor = joint_lr_factor

        self.pred_n_hid = pred_n_hid

        pre_rnn_input_size = in_feats

        post_rnn_input_size = enc_stack_time_factor * enc_n_hid

        enc_mod = {}
        enc_mod["pre_rnn"] = rnn(input_size=pre_rnn_input_size,
                                 hidden_size=enc_n_hid,
                                 num_layers=enc_pre_rnn_layers,
                                 forget_gate_bias=forget_gate_bias,
                                 hidden_hidden_bias_scale=hidden_hidden_bias_scale,
                                 weights_init_scale=weights_init_scale,
                                 dropout=enc_dropout,
                                 decoupled=decoupled_rnns,
                                 tensor_name='pre_rnn',
                                )

        enc_mod["stack_time"] = StackTime(enc_stack_time_factor)

        enc_mod["post_rnn"] = rnn(input_size=post_rnn_input_size,
                                  hidden_size=enc_n_hid,
                                  num_layers=enc_post_rnn_layers,
                                  forget_gate_bias=forget_gate_bias,
                                  hidden_hidden_bias_scale=hidden_hidden_bias_scale,
                                  weights_init_scale=weights_init_scale,
                                  dropout=enc_dropout,
                                  decoupled=decoupled_rnns,
                                  tensor_name='post_rnn',
                                )

        self.encoder = torch.nn.ModuleDict(enc_mod)

        pred_embed = torch.nn.Embedding(n_classes - 1, pred_n_hid)
        logging.log_event(logging.constants.WEIGHTS_INITIALIZATION,
                          metadata=dict(tensor='pred_embed'))
        self.prediction = torch.nn.ModuleDict({
            "embed": pred_embed,
            "dec_rnn": rnn(
                input_size=pred_n_hid,
                hidden_size=pred_n_hid,
                num_layers=pred_rnn_layers,
                forget_gate_bias=forget_gate_bias,
                hidden_hidden_bias_scale=hidden_hidden_bias_scale,
                weights_init_scale=weights_init_scale,
                dropout=pred_dropout,
                decoupled=decoupled_rnns,
                tensor_name='dec_rnn',
            ),
        })

        self.joint_pred = torch.nn.Linear(
            pred_n_hid,
            joint_n_hid)
        logging.log_event(logging.constants.WEIGHTS_INITIALIZATION,
                          metadata=dict(tensor='joint_pred'))
        self.joint_enc = torch.nn.Linear(
            enc_n_hid,
            joint_n_hid)
        logging.log_event(logging.constants.WEIGHTS_INITIALIZATION,
                          metadata=dict(tensor='joint_enc'))

        if apex_mlp:
            # make sure we use the same weight initialization 
            linear_dummy = torch.nn.Linear(joint_n_hid, n_classes)
            fc = MLP([joint_n_hid, n_classes], activation='none')
            with torch.no_grad():
                fc.weights[0].copy_(linear_dummy.weight)
                fc.biases[0].copy_(linear_dummy.bias)
            del linear_dummy
        else:
            fc = torch.nn.Linear(joint_n_hid, n_classes)  

        if fuse_relu_dropout:
            self.joint_net = nn.Sequential(
                FusedReluDropout(joint_dropout),
                fc)
            logging.log_event(logging.constants.WEIGHTS_INITIALIZATION,
                              metadata=dict(tensor='joint_net'))
        else:
            self.joint_net = nn.Sequential(
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(p=joint_dropout),
                fc)
            logging.log_event(logging.constants.WEIGHTS_INITIALIZATION,
                              metadata=dict(tensor='joint_net'))
        self.apex_transducer_joint = apex_transducer_joint

        if self.apex_transducer_joint is not None:
            self.my_transducer_joint= TransducerJoint(
                                        pack_output=(self.apex_transducer_joint=='pack'))
        self.min_lstm_bs = min_lstm_bs

    def forward(self, x, x_lens, y, y_lens, dict_meta_data=None, state=None):
        # x: (B, channels, features, seq_len)
        y = label_collate(y)

        f, x_lens = self.encode(x, x_lens)

        g, _ = self.predict(y, state)
        out = self.joint(f, g, self.apex_transducer_joint, x_lens, dict_meta_data)


        return out, x_lens

    def enc_pred(self, x, x_lens, y, y_lens, pred_stream, state=None):
        pred_stream.wait_stream(torch.cuda.current_stream())
        f, x_lens = self.encode(x, x_lens)

        with torch.cuda.stream(pred_stream):
            y = label_collate(y)
            g, _ = self.predict(y, state)

        torch.cuda.current_stream().wait_stream(pred_stream)
        return f, g, x_lens

    def _seq_merge(self, x):
        # This function assumes input is a list containing 2 seq segments
        assert len(x) == 2, "Only two segment seq split is supprorted now"
        x1_pad = torch.nn.functional.pad(x[1], (0, 0, 0, x[0].size(1)-x[1].size(1)))
        y = torch.cat((x[0], x1_pad), dim=0)
        return y

    def encode(self, x, x_lens):
        """
        Args:
            x: tuple of ``(input, input_lens)``. ``input`` has shape (T, B, I),
                ``input_lens`` has shape ``(B,)``.

        Returns:
            f: tuple of ``(output, output_lens)``. ``output`` has shape
                (B, T, H), ``output_lens``
        """
        require_padding = type(x) is not list and x.size(1) < self.min_lstm_bs
        if require_padding:
            bs = x.size(1)
            x = torch.nn.functional.pad(x, (0, 0, 0, self.min_lstm_bs - bs))
        x, _ = self.encoder["pre_rnn"](x, None)
        x, x_lens = self.encoder["stack_time"](x, x_lens)
        x, _ = self.encoder["post_rnn"](x, None)
        if type(x) is list:
            x = self._seq_merge(x)

        if require_padding:
            x = x[:, :bs]

        f = self.joint_enc(x.transpose(0, 1))

        return f, x_lens

    def predict(self, y, state=None, add_sos=True):
        """
        B - batch size
        U - label length
        H - Hidden dimension size
        L - Number of decoder layers = 2

        Args:
            y: (B, U)

        Returns:
            Tuple (g, hid) where:
                g: (B, U + 1, H)
                hid: (h, c) where h is the final sequence hidden state and c is
                    the final cell state:
                        h (tensor), shape (L, B, H)
                        c (tensor), shape (L, B, H)
        """
        if y is not None:
            # (B, U) -> (B, U, H)
            y = self.prediction["embed"](y)
        else:
            B = 1 if state is None else state[0].size(1)
            y = torch.zeros((B, 1, self.pred_n_hid)).to(
                device=self.joint_enc.weight.device,
                dtype=self.joint_enc.weight.dtype
            )

        # preprend blank "start of sequence" symbol
        if add_sos:
            y = torch.nn.functional.pad(y, (0, 0, 1, 0))

        y = y.transpose(0, 1)#.contiguous()   # (U + 1, B, H)

        bs = y.size(1)
        require_padding = bs < self.min_lstm_bs
        if require_padding:
            y = torch.nn.functional.pad(y, (0, 0, 0, self.min_lstm_bs - bs))

        g, hid = self.prediction["dec_rnn"](y, state)
        g = g.transpose(0, 1)

        if require_padding:
            g = g[:bs]

        g = self.joint_pred(g)

        del y, state
        return g, hid

    def predict_batch(self, y, state=None, add_sos=True):
        """
        B - batch size
        U - label length
        H - Hidden dimension size
        L - Number of decoder layers = 2

        Args:
            y: (B, U)

        Returns:
            Tuple (g, hid) where:
                g: (B, U + 1, H)
                hid: (h, c) where h is the final sequence hidden state and c is
                    the final cell state:
                        h (tensor), shape (L, B, H)
                        c (tensor), shape (L, B, H)
        """
        y_embed = self.prediction["embed"](abs(y.unsqueeze(1)))   # y: [B], y_embed: [B, 1, H]
        mask = (y == -1)
        mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.pred_n_hid)
        y_embed_masked = y_embed.masked_fill_(mask, 0)

        y_embed_masked = y_embed_masked.transpose(0, 1)#.contiguous()   # (U + 1, B, H)
        g, hid = self.prediction["dec_rnn"](y_embed_masked, state)
        g = self.joint_pred(g.transpose(0, 1))
        return g, hid

    def joint(self, f, g, apex_transducer_joint=None, f_len=None, dict_meta_data=None):
        """
        f should be shape (B, T, H)
        g should be shape (B, U + 1, H)

        returns:
            logits of shape (B, T, U, K + 1)
        """
        # Combine the input states and the output states

        if apex_transducer_joint is None:
            f = f.unsqueeze(dim=2)   # (B, T, 1, H)
            g = g.unsqueeze(dim=1)   # (B, 1, U + 1, H)
            h = f + g

            B, T, U, H = h.size()
            res = self.joint_net(h.view(-1, H))
            res = res.view(B, T, U, -1)
        else:
            h = self.my_transducer_joint(f, g, f_len, dict_meta_data["g_len"], 
                                            dict_meta_data["batch_offset"], 
                                            dict_meta_data["packed_batch"])  
            res = self.joint_net(h)

        del f, g
        return res

    def param_groups(self, lr):
        chain_params = lambda *layers: chain(*[l.parameters() for l in layers])
        return [{'params': chain_params(self.encoder),
                 'lr': lr * self.enc_lr_factor},
                {'params': chain_params(self.prediction),
                 'lr': lr * self.pred_lr_factor},
                {'params': chain_params(self.joint_enc, self.joint_pred, self.joint_net),
                 'lr': lr * self.joint_lr_factor},
               ]

class RNNTEncode(nn.Module):
    def __init__(self, encoder, joint_enc, min_lstm_bs):
        super(RNNTEncode, self).__init__()
        self.encoder = encoder
        self.joint_enc = joint_enc
        self.min_lstm_bs = min_lstm_bs

    def forward(self, x, x_lens):
        bs = x.size(1)
        require_padding = bs < self.min_lstm_bs
        if require_padding:
            x = torch.nn.functional.pad(x, (0, 0, 0, self.min_lstm_bs - bs))

        x, _ = self.encoder["pre_rnn"](x, None)
        x, x_lens = self.encoder["stack_time"](x, x_lens)
        x, _ = self.encoder["post_rnn"](x, None)

        if require_padding:
            x = x[:, :bs]

        f = self.joint_enc(x.transpose(0, 1))

        return f, x_lens

class RNNTPredict(nn.Module):
    def __init__(self, prediction, joint_pred, min_lstm_bs):
        super(RNNTPredict, self).__init__()
        self.prediction = prediction
        self.joint_pred = joint_pred
        self.min_lstm_bs = min_lstm_bs

    def forward(self, y):
        y = self.prediction["embed"](y)

        # preprend blank "start of sequence" symbol
        y = torch.nn.functional.pad(y, (0, 0, 1, 0))
        y = y.transpose(0, 1)#.contiguous()   # (U + 1, B, H)

        bs = y.size(1)
        require_padding = bs < self.min_lstm_bs
        if require_padding:
            y = torch.nn.functional.pad(y, (0, 0, 0, self.min_lstm_bs - bs))

        g, hid = self.prediction["dec_rnn"](y, None)
        g = self.joint_pred(g.transpose(0, 1))

        if require_padding:
            g = g[:bs]
        
        return g

def label_collate(labels):
    """Collates the label inputs for the rnn-t prediction network.

    If `labels` is already in torch.Tensor form this is a no-op.

    Args:
        labels: A torch.Tensor List of label indexes or a torch.Tensor.

    Returns:
        A padded torch.Tensor of shape (batch, max_seq_len).
    """

    if isinstance(labels, torch.Tensor):
        return labels.type(torch.int64)
    if not isinstance(labels, (list, tuple)):
        raise ValueError(
            f"`labels` should be a list or tensor not {type(labels)}"
        )

    batch_size = len(labels)
    max_len = max(len(l) for l in labels)

    cat_labels = np.full((batch_size, max_len), fill_value=0.0, dtype=np.int32)
    for e, l in enumerate(labels):
        cat_labels[e, :len(l)] = l
    labels = torch.LongTensor(cat_labels)

    return labels
