# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2019 NVIDIA CORPORATION. All rights reserved.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from maskrcnn_benchmark import _C
from maskrcnn_benchmark import NHWC

from apex import amp

class _FLBROIAlign(Function):
    @staticmethod
    def forward(ctx, input_0, input_1, input_2, input_3, rois, rois_counts, level, output_size, spatial_scale, sampling_ratio, is_nhwc):
        level = level.int()
        ctx.save_for_backward(rois, rois_counts, level)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.is_nhwc = is_nhwc
        ctx.input_0_shape = input_0.size()
        ctx.input_1_shape = input_1.size()
        ctx.input_2_shape = input_2.size()
        ctx.input_3_shape = input_3.size()
        output = _C.flb_roi_align_forward(
            input_0, input_1, input_2, input_3, rois, rois_counts, level, spatial_scale[0], spatial_scale[1], spatial_scale[2], spatial_scale[3],
            output_size[0], output_size[1], sampling_ratio, is_nhwc);
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois,rois_counts,level = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        is_nhwc = ctx.is_nhwc
        if not ctx.is_nhwc:
            bs0, ch0, h0, w0 = ctx.input_0_shape
            bs1, ch1, h1, w1 = ctx.input_1_shape
            bs2, ch2, h2, w2 = ctx.input_2_shape
            bs3, ch3, h3, w3 = ctx.input_3_shape
        else:
            bs0, h0, w0, ch0 = ctx.input_0_shape
            bs1, h1, w1, ch1 = ctx.input_1_shape
            bs2, h2, w2, ch2 = ctx.input_2_shape
            bs3, h3, w3, ch3 = ctx.input_3_shape
        ## TODO: NHWC kernel + transposes is faster than NCHW backward kernel
        ## Might change to transposes + NHWC kernel if we want to speed up NCHW case
        ## Cast to fp32 for the kernel because FP16 atomics is slower than FP32 in Volta
        grad_inputs = _C.flb_roi_align_backward(
            grad_output.float(), rois_counts, rois, level,
            spatial_scale[0], spatial_scale[1], spatial_scale[2], spatial_scale[3],
            output_size[0], output_size[1], bs0, ch0, h0, h1, h2, h3, w0, w1, w2, w3,
            sampling_ratio, is_nhwc)
        return grad_inputs[0].half(), grad_inputs[1].half(), grad_inputs[2].half(), grad_inputs[3].half(), None, None, None, None, None, None, None


flb_roi_align = _FLBROIAlign.apply

class FLBROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio, is_nhwc):
        super(FLBROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.nhwc = is_nhwc

    def forward(self, input_0, input_1, input_2, input_3, rois, rois_counts, level):
        if rois_counts is None:
            rois_counts = torch.tensor([], device=rois.device, dtype=torch.int32)
        return flb_roi_align(
            input_0, input_1, input_2, input_3, rois.float(), rois_counts, level.int(), self.output_size, self.spatial_scale, self.sampling_ratio, self.nhwc
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ")"
        return tmpstr
