import random
import numpy as np
import scipy.ndimage

import mxnet as mx
import mxnet.ndarray as nd
from mxnet.gluon.data.vision import transforms


def get_transforms(shape, layout: str, mode: str, oversampling: float):
    if mode == "train":
        transform_list = [RandBalancedCrop(patch_size=shape, oversampling=oversampling, layout=layout),
                          RandFlip(),
                          RandomBrightnessAugmentation(factor=0.3, prob=0.1),
                          RandomGaussianNoise(mean=0.0, std=0.1, prob=0.1),
                          ToNDArray()]
    elif mode == "validation":
        transform_list = [ToNDArray()]
    else:
        raise ValueError
    return transforms.Compose(transform_list)


class RandBalancedCrop:
    def __init__(self, patch_size, oversampling, layout):
        self.patch_size = patch_size
        self.oversampling = oversampling
        self.layout = layout
        self.axis = [1, 2, 3] if self.layout == "NCDHW" else [0, 1, 2]

    def __call__(self, data):
        image, label = data["image"], data["label"]
        if random.random() < self.oversampling:
            image, label, cords = self.rand_foreg_cropd(image, label)
        else:
            image, label, cords = self._rand_crop(image, label)
        data.update({"image": image, "label": label})
        return data

    @staticmethod
    def randrange(max_range):
        return 0 if max_range == 0 else random.randrange(max_range)

    def get_cords(self, cord, idx):
        return cord[idx], cord[idx] + self.patch_size[idx]

    def _rand_crop(self, image, label):
        image_shape = image.shape[1:] if self.layout == "NCDHW" else image.shape[:-1]
        ranges = [s - p for s, p in zip(image_shape, self.patch_size)]
        cord = [self.randrange(x) for x in ranges]
        low_x, high_x = self.get_cords(cord, 0)
        low_y, high_y = self.get_cords(cord, 1)
        low_z, high_z = self.get_cords(cord, 2)
        if self.layout == "NCDHW":
            image = image[:, low_x:high_x, low_y:high_y, low_z:high_z]
            label = label[:, low_x:high_x, low_y:high_y, low_z:high_z]
        elif self.layout == "NDHWC":
            image = image[low_x:high_x, low_y:high_y, low_z:high_z, :]
            label = label[low_x:high_x, low_y:high_y, low_z:high_z, :]
        else:
            raise ValueError(f"Invalid layout {self.layout}.")
        return image, label, [low_x, high_x, low_y, high_y, low_z, high_z]

    def rand_foreg_cropd(self, image, label):
        def adjust(foreg_slice, patch_size, label, idx):
            diff = patch_size[idx - 1] - (foreg_slice[idx].stop - foreg_slice[idx].start)
            sign = -1 if diff < 0 else 1
            diff = abs(diff)
            ladj = self.randrange(diff)
            hadj = diff - ladj
            low = max(0, foreg_slice[idx].start - sign * ladj)
            high = min(label.shape[idx], foreg_slice[idx].stop + sign * hadj)
            diff = patch_size[idx - 1] - (high - low)
            if diff > 0 and low == 0:
                high += diff
            elif diff > 0:
                low -= diff
            return low, high

        cl = np.random.choice(np.unique(label[label > 0]))
        foreg_slices = scipy.ndimage.find_objects(scipy.ndimage.measurements.label(label==cl)[0])
        foreg_slices = [x for x in foreg_slices if x is not None]
        slice_volumes = [np.prod([s.stop - s.start for s in sl]) for sl in foreg_slices]
        slice_idx = np.argsort(slice_volumes)[-2:]
        foreg_slices = [foreg_slices[i] for i in slice_idx]
        if not foreg_slices:
            return self._rand_crop(image, label)
        foreg_slice = foreg_slices[random.randrange(len(foreg_slices))]
        low_x, high_x = adjust(foreg_slice, self.patch_size, label, self.axis[0])
        low_y, high_y = adjust(foreg_slice, self.patch_size, label, self.axis[1])
        low_z, high_z = adjust(foreg_slice, self.patch_size, label, self.axis[2])
        if self.layout == "NCDHW":
            image = image[:, low_x:high_x, low_y:high_y, low_z:high_z]
            label = label[:, low_x:high_x, low_y:high_y, low_z:high_z]
        elif self.layout == "NDHWC":
            image = image[low_x:high_x, low_y:high_y, low_z:high_z, :]
            label = label[low_x:high_x, low_y:high_y, low_z:high_z, :]
        else:
            raise ValueError(f"Invalid layout {self.layout}.")
        return image, label, [low_x, high_x, low_y, high_y, low_z, high_z]


class RandFlip(mx.gluon.Block):
    def __init__(self, layout="NCDHW"):
        super().__init__()
        self.axis = [1, 2, 3] if layout == "NCDHW" else [0, 1, 2]
        self.prob = 1 / len(self.axis)

    def flip(self, data, axis):
        data["image"] = np.flip(data["image"], axis=axis).copy()
        data["label"] = np.flip(data["label"], axis=axis).copy()
        return data

    def forward(self, data):
        for axis in self.axis:
            if random.random() < self.prob:
                data = self.flip(data, axis)
        return data

    def __call__(self, data):
        return self.forward(data)


class ToNDArray(mx.gluon.Block):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        data["image"] = nd.array(data["image"], dtype=data["image"].dtype)
        data["label"] = nd.array(data["label"], dtype=data["label"].dtype)
        return data

    def __call__(self, data):
        return self.forward(data)


class RandomBrightnessCorrection(mx.gluon.Block): # pylint: disable=R0903
    """ Random brightness correction over samples """
    def __init__(self, factor, prob):
        super().__init__()
        self.prob = prob
        self.factor = factor

    def forward(self, data):
        image = data["image"]
        if random.random() < self.prob:
            factor = mx.nd.random.uniform(low=1.0-self.factor, high=1.0+self.factor, shape=1)
            image = (image * (1 + factor)).astype(image.dtype)
            data.update({"image": image})
        return data


class RandomBrightnessAugmentation(mx.gluon.Block):
    def __init__(self, factor, prob):
        super().__init__()
        self.prob = prob
        self.factor = factor

    def forward(self, data):
        image = data["image"]
        if random.random() < self.prob:
            factor = np.random.uniform(low=1.0-self.factor, high=1.0+self.factor, size=1)
            image = (image * (1 + factor)).astype(image.dtype)
            data.update({"image": image})
        return data


class RandomGaussianNoise(mx.gluon.Block):
    def __init__(self, mean, std, prob):
        super().__init__()
        self.mean = mean
        self.std = std
        self.prob = prob

    def forward(self, data):
        image = data["image"]
        if random.random() < self.prob:
            scale = np.random.uniform(low=0.0, high=self.std)
            noise = np.random.normal(loc=self.mean, scale=scale, size=image.shape).astype(image.dtype)
            data.update({"image": image + noise})
        return data


class RandomGammaCorrection(mx.gluon.Block): # pylint: disable=R0903
    """ Random gamma correction over samples """
    def __init__(self, gamma_range=(0.8, 1.5), keep_stats=False, threshold=0.5, epsilon=1e-8):
        super().__init__()
        self._gamma_range = gamma_range
        self._keep_stats = keep_stats
        self._eps = epsilon
        self._threshold = threshold

    def forward(self, data):
        if random.random() < self._threshold:
            gamma = mx.random.uniform(low=self._gamma_range[0], high=self._gamma_range[1], dtype=data["image"].dtype)
            x_min = data["image"].min()
            x_range = data["image"].max() - x_min
            data["image"] = nd.power((data['image'] - x_min) / (x_range + self._eps), gamma) * x_range + x_min

        return data


class CenterCrop3D(mx.gluon.Block):
    def __init__(self, shape=(128, 128, 128), layout="NCDHW"):
        super().__init__()
        self.shape = shape
        self.layout = layout

    def __call__(self, data):
        image, label = data["image"], data["label"]
        shape = image.shape[1:]
        delta = [max((shape[i] - self.shape[i]) // 2, 0) for i in range(len(self.shape))]
        image = image[
                :,
                delta[0]:delta[0] + self.shape[0],
                delta[1]:delta[1] + self.shape[1],
                delta[2]:delta[2] + self.shape[2]
                ]
        if label is None:
            data.update({"image": image, "label": label})
            return data
        label = label[
                :,
                delta[0]:delta[0] + self.shape[0],
                delta[1]:delta[1] + self.shape[1],
                delta[2]:delta[2] + self.shape[2]
                ]
        data.update({"image": image, "label": label})
        return data


class CenterPad3D(mx.gluon.Block):
    def __init__(self, shape=(128, 128, 128), layout="NCDHW"):
        super().__init__()
        self.shape = shape
        self.layout = layout

    def __call__(self, data):
        image, label = data["image"], data["label"]
        shape = image.shape[1:]
        n_channels = image.shape[0]
        new_shape = tuple([max(shape[i], self.shape[i]) for i in range(len(shape))])
        new_image = nd.zeros(shape=(n_channels,) + new_shape, ctx=image.ctx, dtype=image.dtype)
        new_label = nd.zeros(shape=(n_channels,) + new_shape, ctx=label.ctx, dtype=label.dtype)
        delta = [abs(min((shape[i] - self.shape[i]) // 2, 0)) for i in range(len(self.shape))]
        new_image[
                :,
                delta[0]:delta[0] + shape[0],
                delta[1]:delta[1] + shape[1],
                delta[2]:delta[2] + shape[2]
        ] = image
        new_label[
                :,
                delta[0]:delta[0] + shape[0],
                delta[1]:delta[1] + shape[1],
                delta[2]:delta[2] + shape[2]
                ] = label
        data.update({"image": new_image, "label": new_label})
        return data


class RandomCrop3D(mx.gluon.Block):  # pylint: disable=R0903
    """ Produce a random 3D crop """

    def __init__(self, shape, layout: str = "NCDHW", margins=(0, 0, 0)):
        """ Create op

        :param shape: Target shape
        :param margins: Margins within to perform the crop
        """
        super().__init__()
        self.shape = shape
        self.margins = margins
        self.layout = layout

    def forward(self, data):
        """ Run op
        :param data: dict containing "image" and "label" arrays
        :return: Cropped samples and labels
        """
        image, label = data["image"], data["label"]
        if self.layout == "NCDHW":
            shape = image.shape[1:]
        elif self.layout == "NDHWC":
            shape = image.shape[:-1]
        else:
            raise ValueError("Invalid layout {}".format(self.layout))
        min_ = self.margins
        max_ = [shape[0] - self.shape[0] - self.margins[0],
                shape[1] - self.shape[1] - self.margins[1],
                shape[2] - self.shape[2] - self.margins[2]]
        center = [random.randint(min_[i], max_[i]) for i in range(len(shape))]
        if self.layout == "NCDHW":
            image = image[
                    :,
                    center[0]:center[0] + self.shape[0],
                    center[1]:center[1] + self.shape[1],
                    center[2]:center[2] + self.shape[2]
                    ]
            label = label[
                    :,
                    center[0]:center[0] + self.shape[0],
                    center[1]:center[1] + self.shape[1],
                    center[2]:center[2] + self.shape[2]
                    ]
        else:
            image = image[
                    center[0]:center[0] + self.shape[0],
                    center[1]:center[1] + self.shape[1],
                    center[2]:center[2] + self.shape[2]
                    ]
            label = label[
                    center[0]:center[0] + self.shape[0],
                    center[1]:center[1] + self.shape[1],
                    center[2]:center[2] + self.shape[2]
                    ]

        data.update({"image": image, "label": label})
        return data

    def __call__(self, data):
        return self.forward(data)


# def pad_input(volume, roi_shape, strides, padding_mode, padding_val, dim=3, layout="NCDHW"):
#     """
#     mode: constant, reflect, replicate, circular
#     """
#     image_shape = volume.shape[1:] if layout == "NCDHW" else volume.shape[:-1]
#     bounds = [(strides[i] - image_shape[i] % strides[i]) % strides[i] for i in range(dim)]
#     bounds = [bounds[i] if (image_shape[i] + bounds[i]) >= roi_shape[i] else bounds[i] + strides[i]
#               for i in range(dim)]
#     paddings = [(bounds[i] // 2, bounds[i] - bounds[i] // 2) for i in range(dim)]
#     extra_paddings = [(0, 0)]
#     if layout == "NCDHW":
#         extra_paddings.extend(paddings)
#         paddings = extra_paddings
#     else:
#         paddings.extend(extra_paddings)
#
#     return np.pad(volume, pad_width=paddings, mode=padding_mode, constant_values=padding_val), paddings
#
#
# class WarValCrop(mx.gluon.Block):
#     def __init__(self, roi_shape=(128, 128, 128), layout="NCDHW"):
#         super().__init__()
#         self.roi_shape = roi_shape
#         self.layout = layout
#         self.overlap = 0.5
#
#     def __call__(self, data):
#         inputs, label = data["image"], data["label"]
#         image_shape = list(inputs.shape[1:]) if self.layout == "NCDHW" else list(inputs.shape[:-1])
#         dim = len(image_shape)
#         strides = [int(self.roi_shape[i] * (1 - self.overlap)) for i in range(dim)]
#         bounds = [image_shape[i] % strides[i] for i in range(dim)]
#         bounds = [bounds[i] if bounds[i] < strides[i] // 2 else 0 for i in range(dim)]
#         if self.layout == "NCDHW":
#             inputs = inputs[...,
#                      bounds[0] // 2: image_shape[0] - (bounds[0] - bounds[0] // 2),
#                      bounds[1] // 2: image_shape[1] - (bounds[1] - bounds[1] // 2),
#                      bounds[2] // 2: image_shape[2] - (bounds[2] - bounds[2] // 2)]
#             label = label[...,
#                     bounds[0] // 2: image_shape[0] - (bounds[0] - bounds[0] // 2),
#                     bounds[1] // 2: image_shape[1] - (bounds[1] - bounds[1] // 2),
#                     bounds[2] // 2: image_shape[2] - (bounds[2] - bounds[2] // 2)]
#         elif self.layout == "NDHWC":
#             inputs = inputs[
#                      bounds[0] // 2: image_shape[0] - (bounds[0] - bounds[0] // 2),
#                      bounds[1] // 2: image_shape[1] - (bounds[1] - bounds[1] // 2),
#                      bounds[2] // 2: image_shape[2] - (bounds[2] - bounds[2] // 2),
#                      :
#                      ]
#             label = label[
#                     bounds[0] // 2: image_shape[0] - (bounds[0] - bounds[0] // 2),
#                     bounds[1] // 2: image_shape[1] - (bounds[1] - bounds[1] // 2),
#                     bounds[2] // 2: image_shape[2] - (bounds[2] - bounds[2] // 2),
#                     :
#                     ]
#         else:
#             raise RuntimeError
#
#         inputs, paddings = pad_input(inputs, self.roi_shape, strides, "constant", -2.2, layout=self.layout)
#         label, paddings = pad_input(label, self.roi_shape, strides, "constant", 0, layout=self.layout)
#         data.update({"image": inputs, "label": label})
#         return data
