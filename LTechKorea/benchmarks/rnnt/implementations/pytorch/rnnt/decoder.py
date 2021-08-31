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

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import label_collate
import math
import copy
import amp_C

def graph_simple(func_or_module,
                 sample_args,
                 graph_stream=None,
                 warmup_iters=2,
                 warmup_only=False):
    assert isinstance(sample_args, tuple)
    stream = torch.cuda.Stream() if graph_stream is None else graph_stream
    ambient_stream = torch.cuda.current_stream()
    stream.wait_stream(ambient_stream)
    with torch.cuda.stream(stream):
        # warmup iters before capture
        for _ in range(warmup_iters):
            outputs  = func_or_module(*sample_args)

        if warmup_iters > 0:
            del outputs
        # print("Graphing\n", flush=True)
        # Capture forward pass
        fwd_graph = torch.cuda._Graph()
        fwd_graph.capture_begin()
        outputs  = func_or_module(*sample_args)
        fwd_graph.capture_end()

    ambient_stream.wait_stream(stream)
    def functionalized(*inputs):
        with torch.no_grad():
            for i, arg in zip(sample_args, inputs):
                if i.data_ptr() != arg.data_ptr():
                    i.copy_(arg)
        fwd_graph.replay()      
        return outputs
    return functionalized

class RNNTGreedyDecoder:
    """A greedy transducer decoder.

    Args:
        blank_symbol: See `Decoder`.
        model: Model to use for prediction.
        max_symbols_per_step: The maximum number of symbols that can be added
            to a sequence in a single time step; if set to None then there is
            no limit.
        cutoff_prob: Skip to next step in search if current highest character
            probability is less than this.
    """
    def __init__(self, blank_idx, batch_eval_mode, cg_unroll_factor, rnnt_config, amp_level, max_symbols_per_step=30, max_symbol_per_sample=None):
        self.blank_idx = blank_idx
        assert max_symbols_per_step is None or max_symbols_per_step > 0
        self.max_symbols = max_symbols_per_step
        assert max_symbol_per_sample is None or max_symbol_per_sample > 0
        self.max_symbol_per_sample = max_symbol_per_sample
        self._SOS = -1   # start of sequence
        self.rnnt_config = rnnt_config
        self.cg_captured = False
        self.batch_eval_mode = batch_eval_mode
        self.cg_unroll_factor = cg_unroll_factor
        self.model = None
        self.amp_level = amp_level

    def _pred_step(self, model, label, hidden, device):
        if label == self._SOS:
            return model.predict(None, hidden, add_sos=False)

        if label > self.blank_idx:
            label -= 1

        label = label_collate([[label]]).to(device)
        return model.predict(label, hidden, add_sos=False)

    def _pred_step_batch(self, label, hidden, device):
        # if label == self._SOS:
        #     return model.predict(None, hidden, add_sos=False)

        # if label.max() > self.blank_idx:
        #     label -= 1
        #     raise Exception("Nan: I wonder what this is for")

        # label = label_collate([[label]]).to(device)
        return self.model.predict_batch(label, hidden, add_sos=False)

    def _joint_step(self, model, enc, pred, log_normalize=False):
        logits = model.joint(enc, pred)[:, 0, 0, :]

        if log_normalize:
            probs = F.log_softmax(logits, dim=len(logits.shape) - 1)
            return probs
        else:
            return logits

    def _handle_ema_model(self, ema_model):
        # TODO: maybe use something fancier instead of brute-force deepcopy?
        if self.amp_level == 2:
            ema_model_eval = copy.deepcopy(ema_model)
            ema_model_eval.half()
        else:
            ema_model_eval = ema_model
        ema_model_eval.eval()
        return ema_model_eval

    def capture_cg_for_eval(self, ema_model, dict_meta_data):
        if type(self.batch_eval_mode) == str and not self.batch_eval_mode.startswith("cg"):
            raise Exception("CUDA graph for eval should only be captured when batch_eval_mode == cg")
        if self.cg_captured == True:
            raise Exception("CUDA graph for eval has been captured previously")

        with torch.no_grad():
            ema_model_eval = self._handle_ema_model(ema_model)
            self.model = ema_model_eval
            feats = torch.ones(dict_meta_data["batch"], dict_meta_data["max_feat_len"], self.rnnt_config["joint_n_hid"], dtype=torch.float16, device='cuda')
            feat_lens = torch.ones(dict_meta_data["batch"], dtype=torch.int32, device='cuda') * dict_meta_data["max_feat_len"]
            self._capture_cg(feats, feat_lens)

    def update_ema_model_eval(self, ema_model):
        ema_model_eval = self._handle_ema_model(ema_model)
        if type(self.batch_eval_mode) == str and self.batch_eval_mode.startswith("cg"): 
            if self.cg_captured == False:
                raise Exception("CUDA graph for eval should be captured first before updating")
            else:
                overflow_buf = torch.cuda.IntTensor([0])
                amp_C.multi_tensor_scale(65536, overflow_buf, [list(ema_model_eval.parameters()), list(self.model.parameters())], 1.0)

                # for p, p_eval in zip(self.model.parameters(), ema_model_eval.parameters()):
                #     p.copy_(p_eval)
        else:
            self.model = ema_model_eval

    def decode(self, x, out_lens):
        if self.batch_eval_mode is None:
            return self.decode_reference(self.model, x, out_lens)
        else:
            return self.decode_batch(x, out_lens)

    def decode_reference(self, model, x, out_lens):
        """Returns a list of sentences given an input batch.

        Args:
            x: A tensor of size (batch, channels, features, seq_len)
                TODO was (seq_len, batch, in_features).
            out_lens: list of int representing the length of each sequence
                output sequence.

        Returns:
            list containing batch number of sentences (strings).
        """
        model = getattr(model, 'module', model)
        with torch.no_grad():
            # Apply optional preprocessing
            logits, out_lens = model.encode(x, out_lens)
            output = []
            for batch_idx in range(logits.size(0)):
                inseq = logits[batch_idx, :, :].unsqueeze(1)
                logitlen = out_lens[batch_idx]
                sentence = self._greedy_decode(model, inseq, logitlen)
                output.append(sentence)

        return output

    def decode_batch(self, x, out_lens):
        """Returns a list of sentences given an input batch.

        Args:
            x: A tensor of size (batch, channels, features, seq_len)
                TODO was (seq_len, batch, in_features).
            out_lens: list of int representing the length of each sequence
                output sequence.

        Returns:
            list containing batch number of sentences (strings).
        """
        def next_multiple_of_eight(x):
            return math.ceil(x/8) * 8
        assert self.max_symbol_per_sample is not None, "max_symbol_per_sample needs to be specified in order to use batch_eval"
        with torch.no_grad():
            # Apply optional preprocessing
            B = x.size(1)
            if B < 128:
                B_pad = next_multiple_of_eight(B)
                x = torch.nn.functional.pad(x, (0, 0, 0, B_pad - B))
            logits, out_lens = self.model.encode(x, out_lens)
            logits = logits[:B]

            output = []
            if self.batch_eval_mode == "cg":
                output = self._greedy_decode_batch_replay(logits, out_lens)
            elif self.batch_eval_mode == "cg_unroll_pipeline":
                output = self._greedy_decode_batch_replay_pipelined(logits, out_lens)
            elif self.batch_eval_mode == "no_cg":
                output = self._greedy_decode_batch(logits, out_lens)
        return output

    def _eval_main_loop(self, label_tensor, hidden0, hidden1, time_idx, label_idx, complete_mask, num_symbol_added, num_total_symbol, current_label):
        x, out_len, arange_tensor = self.stashed_tensor
        hidden = [hidden0, hidden1]
        time_idx_clapped = time_idx.data.clone()
        time_idx_clapped.masked_fill_(complete_mask, 0)
        # f is encoder output for this time step
        f = x[arange_tensor, time_idx_clapped, :].unsqueeze(1)

        g, hidden_prime = self.model.predict_batch(current_label, hidden, add_sos=False)

        logp = self.model.joint(f, g)[:, 0, 0, :]
        # get index k, of max prob
        v, k = logp.max(1)
        k = k.int()
        non_blank_mask = (k != self.blank_idx)

        # update current label according to non_blank_mask
        current_label = current_label * ~non_blank_mask + k * non_blank_mask

        label_tensor[arange_tensor, label_idx] = label_tensor[arange_tensor, label_idx] * complete_mask + current_label * ~complete_mask
        for i in range(2):
            expand_mask = non_blank_mask.unsqueeze(0).unsqueeze(2).expand(hidden[0].size())
            hidden[i] = hidden[i] * ~expand_mask + hidden_prime[i] * expand_mask
        # advance time_idx as needed
        num_symbol_added += non_blank_mask
        num_total_symbol += non_blank_mask
        
        time_out_mask = num_total_symbol >= self.max_symbol_per_sample

        exceed_mask = num_symbol_added >= self.max_symbols
        advance_mask = (~non_blank_mask | exceed_mask)  & ~complete_mask
        time_idx += advance_mask
        label_idx += non_blank_mask & ~time_out_mask
        num_symbol_added.masked_fill_(advance_mask, 0)
        
        complete_mask = (time_idx >= out_len) | time_out_mask
        batch_complete = complete_mask.all()

        return label_tensor, hidden[0], hidden[1], time_idx, label_idx, complete_mask, batch_complete, num_symbol_added, num_total_symbol, current_label

    def _eval_main_loop_stream(self, label_tensor, hidden0, hidden1, time_idx, label_idx, complete_mask, num_symbol_added, num_total_symbol, current_label):
        x, out_len, arange_tensor = self.stashed_tensor
        hidden = [hidden0, hidden1]
        time_idx_clapped = time_idx.data.clone()
        time_idx_clapped.masked_fill_(complete_mask, 0)
        # f is encoder output for this time step
        f = x[arange_tensor, time_idx_clapped, :].unsqueeze(1)

        g, hidden_prime = self.model.predict_batch(current_label, hidden, add_sos=False)

        logp = self.model.joint(f, g)[:, 0, 0, :]
        # get index k, of max prob
        v, k = logp.max(1)
        k = k.int()
        non_blank_mask = (k != self.blank_idx)

        # update current label according to non_blank_mask
        self.label_upd_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.label_upd_stream):
            current_label = current_label * ~non_blank_mask + k * non_blank_mask
            label_tensor[arange_tensor, label_idx] = label_tensor[arange_tensor, label_idx] * complete_mask + current_label * ~complete_mask

        self.hidden_upd_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.hidden_upd_stream):
            for i in range(2):
                expand_mask = non_blank_mask.unsqueeze(0).unsqueeze(2).expand(hidden[0].size())
                hidden[i] = hidden[i] * ~expand_mask + hidden_prime[i] * expand_mask
        
        # advance time_idx as needed
        self.time_idx_upd_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.time_idx_upd_stream):
            num_symbol_added += non_blank_mask
            exceed_mask = num_symbol_added >= self.max_symbols
            advance_mask = (~non_blank_mask | exceed_mask)  & ~complete_mask
            time_idx += advance_mask
            num_symbol_added.masked_fill_(advance_mask, 0)

        # handle time out
        num_total_symbol += non_blank_mask
        time_out_mask = num_total_symbol >= self.max_symbol_per_sample
        
        torch.cuda.current_stream().wait_stream(self.label_upd_stream)
        torch.cuda.current_stream().wait_stream(self.hidden_upd_stream)
        torch.cuda.current_stream().wait_stream(self.time_idx_upd_stream)

        label_idx += non_blank_mask & ~time_out_mask
        complete_mask = (time_idx >= out_len) | time_out_mask
        batch_complete = complete_mask.all()

        return label_tensor, hidden[0], hidden[1], time_idx, label_idx, complete_mask, batch_complete, num_symbol_added, num_total_symbol, current_label

    def _eval_main_loop_unroll(self, label_tensor, hidden0, hidden1, time_idx, label_idx, complete_mask, num_symbol_added, num_total_symbol, current_label):
        for u in range(self.cg_unroll_factor):
            label_tensor, hidden0, hidden1, time_idx, label_idx, complete_mask, batch_complete, num_symbol_added, num_total_symbol, \
            current_label = self._eval_main_loop_stream(label_tensor, hidden0, hidden1, time_idx, label_idx, complete_mask, num_symbol_added, num_total_symbol, current_label)

        return label_tensor, hidden0, hidden1, time_idx, label_idx, complete_mask, batch_complete, num_symbol_added, num_total_symbol, current_label


    def _capture_cg_for_main_loop(self, list_input_tensor):
        if self.batch_eval_mode == "cg_unroll_pipeline":
            func_to_be_captured = self._eval_main_loop_unroll
        else:
            func_to_be_captured = self._eval_main_loop_stream
        self.label_upd_stream = torch.cuda.Stream()
        self.hidden_upd_stream = torch.cuda.Stream()
        self.time_idx_upd_stream = torch.cuda.Stream()

        cg = graph_simple(  func_to_be_captured,
                            tuple(t.clone() for t in list_input_tensor),
                            torch.cuda.Stream(),
                            warmup_iters=2)
        return cg

    def _greedy_decode(self, model, x, out_len):
        training_state = model.training
        model.eval()

        device = x.device

        hidden = None
        label = []
        for time_idx in range(out_len):
            if  self.max_symbol_per_sample is not None \
                and len(label) > self.max_symbol_per_sample:
                break
            f = x[time_idx, :, :].unsqueeze(0)

            not_blank = True
            symbols_added = 0

            while not_blank and (
                    self.max_symbols is None or
                    symbols_added < self.max_symbols):
                g, hidden_prime = self._pred_step(
                    model,
                    self._SOS if label == [] else label[-1],
                    hidden,
                    device
                )
                logp = self._joint_step(model, f, g, log_normalize=False)[0, :]

                # get index k, of max prob
                v, k = logp.max(0)
                k = k.item()

                if k == self.blank_idx:
                    not_blank = False
                else:
                    label.append(k)
                    hidden = hidden_prime
                symbols_added += 1

        model.train(training_state)
        return label

    def _greedy_decode_batch(self, x, out_len):
        device = x.device
        B = x.size()[0]

        hidden = None
        assert self.max_symbol_per_sample is not None, "max_symbol_per_sample needs to be specified in order to use batch_eval"
        label_tensor = torch.zeros(B, self.max_symbol_per_sample, dtype=torch.int, device=device)
        current_label = torch.ones(B, dtype=torch.int, device=device) * -1
        
        time_idx = torch.zeros(B, dtype=torch.int64, device=device)
        label_idx = torch.zeros(B, dtype=torch.int64, device=device)
        complete_mask = time_idx >= out_len
        num_symbol_added = torch.zeros(B, dtype=torch.int, device=device)
        num_total_symbol = torch.zeros(B, dtype=torch.int, device=device)
        time_out_mask = torch.ones(B, dtype=torch.bool, device=device)
        arange_tensor = torch.arange(B, device=device)
        while complete_mask.sum().item() != B:
            time_idx_clapped = time_idx.data.clone()
            time_idx_clapped.masked_fill_(complete_mask, 0)
            # f is encoder output for this time step
            f = x[arange_tensor, time_idx_clapped, :].unsqueeze(1)

            ''' The above code is essentially doing '''
            # for i in range(B):
            #     if time_idx[i] < out_len[i]:
            #         f[i, 0, :] = x[i, time_idx[i], :]

            g, hidden_prime = self._pred_step_batch(
                current_label,
                hidden,
                device
            )

            ''' To test the serial joint '''
            # logp = torch.zeros(B, 29, dtype=x.dtype, device=device)
            # for i in range(B):
            #     logp[i, :] = self._joint_step(model, f[i].unsqueeze(0), g[i].unsqueeze(0), log_normalize=False)

            logp = self._joint_step(self.model, f, g, log_normalize=False)
            # get index k, of max prob
            v, k = logp.max(1)
            k = k.int()
            non_blank_mask = (k != self.blank_idx)

            # update current label according to non_blank_mask
            current_label = current_label * ~non_blank_mask + k * non_blank_mask

            if hidden == None:
                hidden = [None, None]
                hidden[0] = torch.zeros_like(hidden_prime[0])
                hidden[1] = torch.zeros_like(hidden_prime[1])
            
            ''' We might need to do the following dynamic resizing '''
            # if (symbol_i >= label_tensor.size(1)):
            #     # resize label tensor 
            #     label_tensor = torch.cat((label_tensor, torch.zeros_like(label_tensor)), dim=1)

            label_tensor[arange_tensor, label_idx] = label_tensor[arange_tensor, label_idx] * complete_mask + current_label * ~complete_mask

            ''' Following is for testing the normal way of generate label '''
            # for i in range(B):
            #     if non_blank_mask[i] and time_idx[i] < out_len[i]:
            #         # pdb.set_trace()
            #         label_ref[i].append(current_label[i].item()) 

            # update hidden if the inference result is non-blank
            for i in range(2):
                expand_mask = non_blank_mask.unsqueeze(0).unsqueeze(2).expand(hidden[0].size())
                hidden[i] = hidden[i] * ~expand_mask + hidden_prime[i] * expand_mask
            # advance time_idx as needed
            num_symbol_added += non_blank_mask
            num_total_symbol += non_blank_mask
            if self.max_symbol_per_sample == None:
                time_out_mask = torch.zeros_like(complete_mask)
            else:
                time_out_mask = num_total_symbol >= self.max_symbol_per_sample

            
            exceed_mask = num_symbol_added >= self.max_symbols
            advance_mask = (~non_blank_mask | exceed_mask)  & ~complete_mask
            time_idx += advance_mask
            label_idx += non_blank_mask & ~time_out_mask
            num_symbol_added.masked_fill_(advance_mask, 0)
            
            complete_mask = (time_idx >= out_len) | time_out_mask
            

        label = []
        
        for i in range(B):
            label.append(label_tensor[i, :label_idx[i]].tolist())
        return label

    def _capture_cg(self, x, out_len):
        self.model.eval()
        device = x.device
        B = x.size()[0]
        self.cg_batch_size = B

        hidden = [torch.zeros((self.rnnt_config["pred_rnn_layers"], B, self.rnnt_config["pred_n_hid"]), 
                                dtype=x.dtype, device=device)]*2
        assert self.max_symbol_per_sample is not None, "max_symbol_per_sample needs to be specified in order to use batch_eval"
        label_tensor = torch.zeros(B, self.max_symbol_per_sample, dtype=torch.int, device=device)
        current_label = torch.ones(B, dtype=torch.int, device=device) * -1
        
        time_idx = torch.zeros(B, dtype=torch.int64, device=device)
        label_idx = torch.zeros(B, dtype=torch.int64, device=device)
        complete_mask = time_idx >= out_len
        num_symbol_added = torch.zeros(B, dtype=torch.int, device=device)
        num_total_symbol = torch.zeros(B, dtype=torch.int, device=device)
        arange_tensor = torch.arange(B, device=device)

        list_input_tensor = [label_tensor, hidden[0], hidden[1], time_idx, label_idx, complete_mask, 
                                num_symbol_added, num_total_symbol, current_label]

        self.stashed_tensor = x, out_len, arange_tensor
        
        self.main_loop_cg = self._capture_cg_for_main_loop(list_input_tensor)
        self.cg_captured = True

    def _greedy_decode_batch_replay(self, x, out_len):
        device = x.device
        B = x.size()[0]
        assert B <= self.cg_batch_size, "this should not have happened"

        # self.stashed_tensor = x, out_len, arange_tensor
        self.stashed_tensor[0][:x.size(0), :x.size(1)] = x
        self.stashed_tensor[1][:out_len.size(0)] = out_len

        hidden = [torch.zeros((2, self.cg_batch_size, self.rnnt_config["pred_n_hid"]), dtype=x.dtype, device=device)]*2
        label_tensor = torch.zeros(self.cg_batch_size, self.max_symbol_per_sample, dtype=torch.int, device=device)
        current_label = torch.ones(self.cg_batch_size, dtype=torch.int, device=device) * -1
        
        time_idx = torch.zeros(self.cg_batch_size, dtype=torch.int64, device=device)
        label_idx = torch.zeros(self.cg_batch_size, dtype=torch.int64, device=device)
        complete_mask = time_idx >= self.stashed_tensor[1] # i.e. padded out_len
        batch_complete = complete_mask.all()
        num_symbol_added = torch.zeros(self.cg_batch_size, dtype=torch.int, device=device)
        num_total_symbol = torch.zeros(self.cg_batch_size, dtype=torch.int, device=device)
        arange_tensor = torch.arange(self.cg_batch_size, device=device)

        list_input_tensor = [label_tensor, hidden[0], hidden[1], time_idx, label_idx, complete_mask, num_symbol_added, num_total_symbol, 
            current_label]

        
        while batch_complete == False:
            list_input_tensor = [label_tensor, hidden[0], hidden[1], time_idx, label_idx, complete_mask, num_symbol_added, num_total_symbol, 
            current_label]
            label_tensor, hidden[0], hidden[1], time_idx, label_idx, complete_mask, batch_complete, num_symbol_added, num_total_symbol, \
            current_label = self.main_loop_cg(*list_input_tensor)

        label = []
        
        for i in range(B):
            label.append(label_tensor[i, :label_idx[i]].tolist())
        return label

    def _greedy_decode_batch_replay_pipelined(self, x, out_len):
        device = x.device
        B = x.size()[0]
        assert B <= self.cg_batch_size, "this should not have happened"

        # self.stashed_tensor = x, out_len, arange_tensor
        self.stashed_tensor[0][:x.size(0), :x.size(1)] = x
        self.stashed_tensor[1][:out_len.size(0)] = out_len

        hidden = [torch.zeros((2, self.cg_batch_size, self.rnnt_config["pred_n_hid"]), dtype=x.dtype, device=device)]*2
        label_tensor = torch.zeros(self.cg_batch_size, self.max_symbol_per_sample, dtype=torch.int, device=device)
        current_label = torch.ones(self.cg_batch_size, dtype=torch.int, device=device) * -1
        
        time_idx = torch.zeros(self.cg_batch_size, dtype=torch.int64, device=device)
        label_idx = torch.zeros(self.cg_batch_size, dtype=torch.int64, device=device)
        complete_mask = time_idx >= self.stashed_tensor[1] # i.e. padded out_len
        batch_complete = complete_mask.all()
        num_symbol_added = torch.zeros(self.cg_batch_size, dtype=torch.int, device=device)
        num_total_symbol = torch.zeros(self.cg_batch_size, dtype=torch.int, device=device)
        arange_tensor = torch.arange(self.cg_batch_size, device=device)

        list_input_tensor = [label_tensor, hidden[0], hidden[1], time_idx, label_idx, complete_mask, num_symbol_added, num_total_symbol, 
            current_label]

        
        batch_complete_cpu = torch.tensor(False, dtype=torch.bool, device='cpu').pin_memory()
        copy_stream = torch.cuda.Stream()
        while True:
            list_input_tensor = [label_tensor, hidden[0], hidden[1], time_idx, label_idx, complete_mask, num_symbol_added, num_total_symbol, 
            current_label]
            label_tensor, hidden[0], hidden[1], time_idx, label_idx, complete_mask, batch_complete, num_symbol_added, num_total_symbol, \
            current_label = self.main_loop_cg(*list_input_tensor)

            copy_stream.synchronize()
            # print(batch_complete_cpu)
            if torch.any(batch_complete_cpu):
                break

            copy_stream.wait_stream(torch.cuda.current_stream())

            with torch.cuda.stream(copy_stream):
                batch_complete_cpu.copy_(batch_complete, non_blocking=True)

        label = []
        
        for i in range(B):
            label.append(label_tensor[i, :label_idx[i]].tolist())
        return label


