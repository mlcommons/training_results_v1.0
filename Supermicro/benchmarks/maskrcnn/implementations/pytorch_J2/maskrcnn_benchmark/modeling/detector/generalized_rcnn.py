# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.layers.nhwc import nchw_to_nhwc_transform, nhwc_to_nchw_transform

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.rpn_head_included_in_backbone = cfg.MODEL.BACKBONE.INCLUDE_RPN_HEAD
        self.precompute_rpn_constant_tensors = cfg.PRECOMPUTE_RPN_CONSTANT_TENSORS
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)
        self.nhwc = cfg.NHWC
        self.dali = cfg.DATALOADER.DALI

    def forward_backbone(self, images):
        if self.rpn_head_included_in_backbone:
            flattened_results = self.backbone(images.tensors)
            num_res = len(flattened_results) // 4
            features, objectness, rpn_box_regression = flattened_results[0:2*num_res], flattened_results[2*num_res:3*num_res], flattened_results[3*num_res:4*num_res]
        else:
            features, objectness, rpn_box_regression = self.backbone(images.tensors), None, None
        features = [tuple(features[0:5]),
                    tuple(features[5:10])]
        return features, objectness, rpn_box_regression

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        if self.nhwc and not self.dali:
            # data-loader outputs nchw images
            images.tensors = nchw_to_nhwc_transform(images.tensors)
        elif self.dali and not self.nhwc:
            # dali pipeline outputs nhwc images
            images.tensors = nhwc_to_nchw_transform(images.tensors)
        features, objectness, rpn_box_regression = self.forward_backbone(images)
        proposals, proposal_losses = self.rpn(images, features, targets, objectness, rpn_box_regression) if not self.nhwc else \
                                     self.rpn(images, features[1], targets, objectness, rpn_box_regression)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets) if not self.nhwc else \
                                         self.roi_heads(features[0], proposals, targets)
        ## for NHWC layout case, features[0] are NHWC features, and [1] NCHW
        else:
            # RPN-only models don't have roi_heads
            ## TODO: take care of NHWC/NCHW cases for RPN-only case 
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
