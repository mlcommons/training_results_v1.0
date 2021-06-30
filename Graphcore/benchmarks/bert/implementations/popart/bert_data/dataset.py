# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

from itertools import chain
import numpy as np


class DataSet(object):
    '''
    Converts an iterator that returns a list of np.ndarrays into an iterator that returns
    the same list with the ndarrays reshaped to match PopART's dataflow requirements.
    '''

    def __init__(self,
                 loader,
                 tensor_shapes,
                 batches_per_step=1,
                 replication_factor=1,
                 accumulation_factor=1):
        self.tensor_shapes = tensor_shapes
        self.loader = loader
        self.batches_per_step = batches_per_step
        self.replication_factor = replication_factor
        self.accumulation_factor = accumulation_factor
        self.steps_per_epoch = len(loader)

        # Determine the shape of the batch based on samples_per_step, accumulation_factor and replication_factor
        self.outer_shapes = []

        # PopART expects inputs to be of the shape [batches_per_step, accl_factor, repl_factor, micro_batch, *data_shape]
        if self.batches_per_step > 1:
            self.outer_shapes += [self.batches_per_step]

        if self.accumulation_factor > 1:
            self.outer_shapes += [self.accumulation_factor]

        if self.replication_factor > 1:
            self.outer_shapes += [self.replication_factor]

    def __iter__(self):
        self.loader_iterator = iter(self.loader)
        return self

    def __len__(self):
        return len(self.loader)

    def __next__(self):
        # Get the next sample/label
        items = next(self.loader_iterator)
        tensor_names = []

        # Reshape the input
        feed_dict = {}
        for i, (id, shape) in enumerate(self.tensor_shapes):
            if shape is not None:
                items[i] = items[i].reshape(tuple(chain(self.outer_shapes, shape)))
                feed_dict[id] = items[i]

        return feed_dict
