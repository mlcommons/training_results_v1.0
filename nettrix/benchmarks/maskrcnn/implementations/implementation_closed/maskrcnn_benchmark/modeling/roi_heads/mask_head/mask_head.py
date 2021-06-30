# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList

from .roi_mask_feature_extractors import make_roi_mask_feature_extractor
from .roi_mask_predictors import make_roi_mask_predictor
from .inference import make_roi_mask_post_processor
from .loss import make_roi_mask_loss_evaluator


def keep_only_positive_boxes(boxes, syncfree, max_pos_inds):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
        syncfree (bool, True if syncfree code path is preferred)
        positive_fraction (float)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    positive_inds_counts = []
    weights = []
    total_count, total_numel = None, 0
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        if syncfree:
            import maskrcnn_benchmark.Syncfree
            inds, counts = maskrcnn_benchmark.Syncfree.nonzero_repeat(inds_mask, torch.tensor([], device=inds_mask.device, dtype=torch.int64))
            inds = inds[:max_pos_inds]
            positive_inds_counts.append(counts)
            w = torch.arange(0,inds.numel(),1, device=inds.device)
            w = w < counts
            weights.append(w)
            total_count = counts if total_count is None else total_count + counts
            total_numel = total_numel + inds.numel()
        else:
            with torch.cuda.nvtx.range("NZ5"):
                inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    if syncfree:
        return positive_boxes, positive_inds, weights, total_numel / total_count, positive_inds_counts
    else:
        return positive_boxes, positive_inds, None, 1.0, None


class ROIMaskHead(torch.nn.Module):
    def __init__(self, cfg):
        super(ROIMaskHead, self).__init__()
        self.cfg = cfg.clone()
        self.max_pos_inds = int(cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE*cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION)
        self.feature_extractor = make_roi_mask_feature_extractor(cfg)
        self.predictor = make_roi_mask_predictor(cfg)
        self.post_processor = make_roi_mask_post_processor(cfg)
        self.loss_evaluator = make_roi_mask_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None, syncfree=False):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        syncfree = syncfree and len(proposals) == 1 # syncfree path is slower when N > 1
        positive_inds_counts = None
        if self.training:
            # during training, only focus on positive boxes
            with torch.no_grad():
                all_proposals = proposals
                proposals, positive_inds, weights, scale, positive_inds_counts = keep_only_positive_boxes(proposals, syncfree, self.max_pos_inds)
                positive_inds_counts = positive_inds_counts[0] if syncfree and len(proposals) == 1 else None
        if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            x = self.feature_extractor(features, proposals, positive_inds_counts)
        mask_logits = self.predictor(x)

        if not self.training:
            result = self.post_processor(mask_logits, proposals)
            return x, result, {}

        loss_mask = self.loss_evaluator(proposals, mask_logits.float(), targets, weights, scale)

        return None if syncfree else x, all_proposals, dict(loss_mask=loss_mask)


def build_roi_mask_head(cfg):
    return ROIMaskHead(cfg)
