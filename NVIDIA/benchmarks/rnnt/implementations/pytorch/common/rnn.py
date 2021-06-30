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

import math

import torch
from torch.nn import Parameter
from mlperf import logging


def rnn(input_size, hidden_size, num_layers,
        forget_gate_bias=1.0, dropout=0.0,
        decoupled=False, **kwargs):

    kwargs = dict(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        forget_gate_bias=forget_gate_bias,
        **kwargs,
    )

    if decoupled:
        return DecoupledLSTM(**kwargs)
    else:
        return LSTM(**kwargs)


class LSTM(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout,
                 forget_gate_bias, weights_init_scale=1.0,
                 hidden_hidden_bias_scale=0.0, **kwargs):
        """Returns an LSTM with forget gate bias init to `forget_gate_bias`.

        Args:
            input_size: See `torch.nn.LSTM`.
            hidden_size: See `torch.nn.LSTM`.
            num_layers: See `torch.nn.LSTM`.
            dropout: See `torch.nn.LSTM`.
            forget_gate_bias: For each layer and each direction, the total value of
                to initialise the forget gate bias to.

        Returns:
            A `torch.nn.LSTM`.
        """
        super(LSTM, self).__init__()

        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.dropout = torch.nn.Dropout(dropout) if dropout else None

        if forget_gate_bias is not None:
            for name, v in self.lstm.named_parameters():
                if "bias_ih" in name:
                    bias = getattr(self.lstm, name)
                    bias.data[hidden_size:2*hidden_size].fill_(forget_gate_bias)
                if "bias_hh" in name:
                    bias = getattr(self.lstm, name)
                    bias.data[hidden_size:2*hidden_size] *= float(hidden_hidden_bias_scale)

        for name, v in self.named_parameters():
            if 'weight' in name or 'bias' in name:
                v.data *= float(weights_init_scale)
        tensor_name = kwargs['tensor_name']
        logging.log_event(logging.constants.WEIGHTS_INITIALIZATION,
                          metadata=dict(tensor=tensor_name))


    def forward(self, x, h=None):
        if type(x) is not list:
            x, h = self.lstm(x, h)
            if self.dropout:
                x = self.dropout(x)
            return x, h
        else:
            # seq splitting path
            if len(x) != 2:
                raise NotImplementedError("Only number of seq segments equal to 2 is supported")
            y0, h0 = self.lstm(x[0], h)
            hid0 = h0[0][:, :x[1].size(1)].contiguous()
            cell0 = h0[1][:, :x[1].size(1)].contiguous()
            y1, h1 = self.lstm(x[1], (hid0, cell0))
            if self.dropout:
                y0 = self.dropout(y0)
                y1 = self.dropout(y1)
            # h will not be used in training any way. Return None.
            # We guarantee this path will only be taken by training
            return [y0, y1], None


class DecoupledBase(torch.nn.Module):
    """Base class for decoupled RNNs.

    Meant for being sub-classed, with children class filling self.rnn
    with RNN cells.
    """
    def __init__(self, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.rnn = torch.nn.ModuleList()

    def forward(self, x, hx=None):
        assert len(self.rnn) > 0, "RNN not initialized"

        hx = self._parse_hidden_state(hx)

        hs = []
        cs = []
        rnn_idx = 0
        for layer in self.rnn:
            if isinstance(layer, torch.nn.Dropout):
                x = layer(x)
            else:
                x, h_out = layer(x, hx[rnn_idx])
                hs.append(h_out[0])
                cs.append(h_out[1])
                rnn_idx += 1
                del h_out

        h_0 = torch.cat(hs, dim=0)
        c_0 = torch.cat(cs, dim=0)
        return x, (h_0, c_0)

    def _parse_hidden_state(self, hx):
        """
        Dealing w. hidden state:
        Typically in pytorch: (h_0, c_0)
            h_0 = ``[num_layers * num_directions, batch, hidden_size]``
            c_0 = ``[num_layers * num_directions, batch, hidden_size]``
        """
        if hx is None:
            return [None] * self.num_layers
        else:
            h_0, c_0 = hx
            assert h_0.shape[0] == self.num_layers
            return [(h_0[i].unsqueeze(dim=0), c_0[i].unsqueeze(dim=0)) for i in range(h_0.shape[0])]

    def _flatten_parameters(self):
        for layer in self.rnn:
            if isinstance(layer, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)):
                layer._flatten_parameters()

    def set_forgate_gate_bias(self, val, hidden_size, hidden_hidden_bias_scale):
        # TODO Check it with debugger for single_layer
        if val is not None:
            for name, v in self.rnn.named_parameters():
                if "bias_ih" in name:
                    idx, name = name.split('.', 1)
                    bias = getattr(self.rnn[int(idx)], name)
                    bias.data[hidden_size:2*hidden_size].fill_(val)
                if "bias_hh" in name:
                    idx, name = name.split('.', 1)
                    bias = getattr(self.rnn[int(idx)], name)
                    bias.data[hidden_size:2*hidden_size] *= float(hidden_hidden_bias_scale)


class DecoupledLSTM(DecoupledBase):
    def __init__(self, input_size, hidden_size, num_layers, dropout,
            hidden_hidden_bias_scale, weights_init_scale,
                 forget_gate_bias, multilayer_cudnn=True, **kwargs):
        super().__init__(num_layers)

        for i in range(num_layers):
            self.rnn.append(torch.nn.LSTM(
                input_size=input_size if i == 0 else hidden_size,
                hidden_size=hidden_size))
            self.rnn.append(torch.nn.Dropout(dropout))

        self.set_forgate_gate_bias(forget_gate_bias, hidden_size, hidden_hidden_bias_scale)

        for name, v in self.named_parameters():
            if 'weight' in name or 'bias' in name:
                v.data *= float(weights_init_scale)

        tensor_name = kwargs['tensor_name']
        logging.log_event(logging.constants.WEIGHTS_INITIALIZATION,
                          metadata=dict(tensor=tensor_name))
