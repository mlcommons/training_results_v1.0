from mxnet.gluon import HybridBlock


class COCOInference(HybridBlock):
    def __init__(self, net, ltrb=False, scale_bboxes=True, score_threshold=0.0, **kwargs):
        '''
        net - SSD network
        ltrb (bool, optional, default = False) – If true, bboxes are returned as [left, top, right, bottom],
                                                 else [x, y, width, height].
        scale_bboxes (bool, optional, default = False) – If False, bboxes returned values as expressed as ratio w.r.t.
                                                         to the image width and height. otherwise, bboxes will be
                                                         scaled based on "scale" input if provided
        '''
        super(COCOInference, self).__init__(**kwargs)
        self.net = net
        self.ltrb = ltrb
        self.scale_bboxes = scale_bboxes
        self.score_threshold = score_threshold

    def hybrid_forward(self, F, images, anchors, scale=None, ids=None):
        labels, scores, bboxes = self.net(images, anchors)
        bboxes = F.clip(bboxes, a_min=0, a_max=1)  # clip normalized bboxes to [0, 1]

        # ids needs to be in fp32 as fp16 doesn't have enough range.
        # which means labels need to be in fp16 too (broadcast_like), and since we
        # concat [bboxes, scores, labels, ids], it's easier if we convert everything to
        # with fp32.
        labels = labels.astype(dtype='float32')
        scores = scores.astype(dtype='float32')
        bboxes = bboxes.astype(dtype='float32')
        if scale is not None:
            scale = scale.astype(dtype='float32')
        if ids is not None:
            ids = ids.astype(dtype='float32')

        # scale bboxes to original image shape
        if self.scale_bboxes or not self.ltrb:
            bboxes0 = F.slice_axis(bboxes, axis=2, begin=0, end=1).flatten()
            bboxes1 = F.slice_axis(bboxes, axis=2, begin=1, end=2).flatten()
            bboxes2 = F.slice_axis(bboxes, axis=2, begin=2, end=3).flatten()
            bboxes3 = F.slice_axis(bboxes, axis=2, begin=3, end=4).flatten()
            if self.scale_bboxes:
                h_scale = F.slice_axis(scale, axis=1, begin=0, end=1)
                w_scale = F.slice_axis(scale, axis=1, begin=1, end=2)
                bboxes0 = F.broadcast_mul(bboxes0, w_scale)
                bboxes1 = F.broadcast_mul(bboxes1, h_scale)
                bboxes2 = F.broadcast_mul(bboxes2, w_scale)
                bboxes3 = F.broadcast_mul(bboxes3, h_scale)
            if not self.ltrb:
                # convert [left, top, right, bottom] to [x, y, width, height].
                bboxes2 = bboxes2 - bboxes0 + 1
                bboxes3 = bboxes3 - bboxes1 + 1
            bboxes = F.stack(*[bboxes0, bboxes1, bboxes2, bboxes3], axis=2)

        if ids is None:
            results = F.concat(*[bboxes, scores, labels], dim=2)
        else:
            # repeat ids to have same shape as labels
            ids = F.expand_dims(ids, axis=1)
            ids = F.broadcast_like(ids, labels)
            # concatenate
            results = F.concat(*[bboxes, scores, labels, ids], dim=2)

        # Get a mask for results with score lower than score_threshold
        # Note that we apply the mask outside this block to allow static shape hybridization
        mask = labels >= self.score_threshold

        return results, mask
