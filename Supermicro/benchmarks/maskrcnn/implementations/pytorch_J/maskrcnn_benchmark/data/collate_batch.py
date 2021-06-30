# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.structures.image_list import to_image_list


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0, shapes=None, passthrough=False):
        self.size_divisible = size_divisible
        self.shapes = shapes
        print(f'shapes={shapes}')
        self.passthrough = passthrough
        print(f'passthrough={passthrough}')

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        if self.passthrough:
            images = transposed_batch[0]
            targets = transposed_batch[1]
            img_ids = transposed_batch[2]
        else:
            images = to_image_list(transposed_batch[0], self.size_divisible, self.shapes)
            targets = transposed_batch[1]
            img_ids = transposed_batch[2]
        return images, targets, img_ids
