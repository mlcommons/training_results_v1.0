import torch
from rnnt.model import RNNT, RNNTEncode, RNNTPredict, label_collate
from rnnt.function import graph

class RNNTGraph():
    def __init__(self, model, rnnt_config, batch_size, max_feat_len, max_txt_len, num_cg):
        self.model = model
        self.rnnt_config = rnnt_config
        self.batch_size = batch_size
        self.cg_stream = torch.cuda.Stream()
        self.encode_stream = torch.cuda.Stream()
        self.predict_stream = torch.cuda.Stream()
        self.max_feat_len = max_feat_len
        self.max_txt_len = max_txt_len
        self.num_cg = num_cg

    def _gen_encode_graph(self, max_feat_len):
        feats = torch.ones(max_feat_len, self.batch_size, self.rnnt_config["in_feats"], dtype=torch.float16, device='cuda')
        feat_lens = torch.ones(self.batch_size, dtype=torch.int32, device='cuda') * max_feat_len

        encode_args = (feats, feat_lens)
        rnnt_encode = RNNTEncode(self.model.encoder, self.model.joint_enc, self.model.min_lstm_bs)
        encode_segment = graph( rnnt_encode,
                                encode_args,
                                self.cg_stream,
                                warmup_iters=2,
                                warmup_only=False)
        return encode_segment

    def _gen_predict_graph(self, max_txt_len):
        txt = torch.ones(self.batch_size, max_txt_len, dtype=torch.int64, device='cuda')
        predict_args = (txt, )
        rnnt_predict = RNNTPredict(self.model.prediction, self.model.joint_pred, self.model.min_lstm_bs)
        predict_segment = graph(rnnt_predict,
                                predict_args,
                                self.cg_stream,
                                warmup_iters=2,
                                warmup_only=False)
        return predict_segment


    def capture_graph(self):
        list_encode_segment = []
        list_predict_segment = []
        list_max_feat_len = []
        list_max_txt_len = []
        for i in range(self.num_cg):
            list_max_feat_len.append(self.max_feat_len - (i*self.max_feat_len//self.num_cg))
            list_encode_segment.append(self._gen_encode_graph(list_max_feat_len[i]))

            list_max_txt_len.append(self.max_txt_len - (i*self.max_txt_len//self.num_cg))
            list_predict_segment.append(self._gen_predict_graph(list_max_txt_len[i]))

        # build a hash table
        self.dict_encode_graph = {}
        curr_list_ptr = len(list_max_feat_len) - 1
        for feat_len in range(1, self.max_feat_len+1):
            while feat_len > list_max_feat_len[curr_list_ptr]:
                curr_list_ptr -= 1
                assert curr_list_ptr >= 0
            self.dict_encode_graph[feat_len] = (list_max_feat_len[curr_list_ptr], list_encode_segment[curr_list_ptr])

        # build a hash table
        self.dict_predict_graph = {}
        curr_list_ptr = len(list_max_txt_len) - 1
        for txt_len in range(1, self.max_txt_len+1):
            while txt_len > list_max_txt_len[curr_list_ptr]:
                curr_list_ptr -= 1
                assert curr_list_ptr >= 0
            self.dict_predict_graph[txt_len] = (list_max_txt_len[curr_list_ptr], list_predict_segment[curr_list_ptr])


    def _model_segment(self, encode_block, predict_block, x, x_lens, y, y_lens, dict_meta_data=None):
        f, x_lens = encode_block(x, x_lens)
        # g, _ = self.model.predict(y)
        g = predict_block(y) 
        out = self.model.joint(f, g, self.model.apex_transducer_joint, x_lens, dict_meta_data)
        return out, x_lens

    def step(self, feats, feat_lens, txt, txt_lens, dict_meta_data):
        max_feat_len, encode_block = self.dict_encode_graph[feats.size(0)]
        max_txt_len, predict_block = self.dict_predict_graph[txt.size(1)]
        assert feats.size(0) <= max_feat_len, "got feat_len of %d" % feats.size(0)
        feats = torch.nn.functional.pad(feats, (0, 0, 0, 0, 0, max_feat_len-feats.size(0)))
        txt = torch.nn.functional.pad(txt, (0, max_txt_len-txt.size(1)))
        log_probs, log_prob_lens = self._model_segment(encode_block, predict_block, feats, feat_lens, txt, txt_lens, dict_meta_data)
        return log_probs, log_prob_lens


