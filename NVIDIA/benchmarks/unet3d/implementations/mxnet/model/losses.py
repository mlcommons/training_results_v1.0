import numpy as np
from mxnet.gluon.loss import Loss, SoftmaxCrossEntropyLoss
import pdb


class DiceCELoss(Loss):
    def __init__(self, to_onehot_y: bool = True, use_softmax: bool = True, include_background: bool = False):
        super(DiceCELoss, self).__init__(weight=None, batch_axis=0)
        self.channel_axis = -1
        self.dice = Dice(to_onehot_y=to_onehot_y, use_softmax=use_softmax, include_background=include_background)
        self.cross_entropy = SoftmaxCrossEntropyLoss(sparse_label=True, axis=self.channel_axis)

    def hybrid_forward(self, F, y_pred, y_true, *args, **kwargs):
        dice = 1.0 - F.mean(self.dice(y_pred, y_true))
        cross_entropy = F.mean(self.cross_entropy(y_pred, F.squeeze(y_true, axis=self.channel_axis)))
        return (dice + cross_entropy) / 2


class DiceScore(Loss):
    def __init__(self, to_onehot_y: bool = True, use_argmax: bool = True, include_background: bool = False):
        super(DiceScore, self).__init__(weight=None, batch_axis=0)
        self.dice = Dice(to_onehot_y=to_onehot_y, to_onehot_x=True, use_softmax=False,
                         use_argmax=use_argmax, include_background=include_background)

    def hybrid_forward(self, F, y_pred, y_true, *args, **kwargs):
        return F.mean(self.dice(y_pred, y_true), axis=0)


class Dice(Loss):
    def __init__(self,
                 to_onehot_y: bool = True,
                 to_onehot_x: bool = False,
                 use_softmax: bool = True,
                 use_argmax: bool = False,
                 include_background: bool = False):
        super(Dice, self).__init__(weight=None, batch_axis=0)
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.to_onehot_x = to_onehot_x
        self.use_softmax = use_softmax
        self.use_argmax = use_argmax
        self.smooth_nr = 1e-6
        self.smooth_dr = 1e-6
        self.cast_type = np.float32

    def cast(self, dtype):
        self.cast_type = np.float16 if ((dtype == 'float16') or (dtype == np.float16)) else np.float32

    def hybrid_forward(self, F, y_pred, y_true, *args, **kwargs):
        channel_axis = -1
        reduce_axis = list(range(1, 4))
        num_pred_ch = 3

        if self.use_softmax:
            y_pred = F.softmax(y_pred, axis=channel_axis)
        elif self.use_argmax:
            y_pred = F.argmax(y_pred, axis=channel_axis, keepdims=True)

        if self.to_onehot_y:
            y_true = self.to_one_hot(F, y_true, channel_axis, num_pred_ch)

        if self.to_onehot_x:
            y_pred = self.to_one_hot(F, y_pred, channel_axis, num_pred_ch)

        if not self.include_background:
            assert num_pred_ch > 1, \
                f"To exclude background the prediction needs more than one channel. Got {num_pred_ch}."
            y_true = F.slice_axis(y_true, axis=-1, begin=1, end=3)
            y_pred = F.slice_axis(y_pred, axis=-1, begin=1, end=3)

        intersection = F.sum(y_true * y_pred, axis=reduce_axis)
        target_sum = F.sum(y_true, axis=reduce_axis)
        prediction_sum = F.sum(y_pred, axis=reduce_axis)

        dice = (2.0 * intersection + self.smooth_nr) / (target_sum + prediction_sum + self.smooth_dr)
        return dice

    def to_one_hot(self, F, array, channel_axis, num_pred_ch):
        # return F.one_hot(F.squeeze(array, axis=channel_axis), depth=num_pred_ch, dtype=self.cast_type)
        return F.one_hot(F.squeeze(array, axis=channel_axis), depth=num_pred_ch)
