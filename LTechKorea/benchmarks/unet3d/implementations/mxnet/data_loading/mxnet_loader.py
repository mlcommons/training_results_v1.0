import numpy as np
from mxnet import gluon, nd

from data_loading.transforms import get_transforms


class KitsDataset(gluon.data.Dataset):
    def __init__(self, flags, image_list: list, label_list: list, mode: str):
        self.image_list = image_list
        self.label_list = label_list
        self.transforms = get_transforms(flags.input_shape if mode == "train" else flags.val_input_shape,
                                         flags.layout, mode=mode, oversampling=flags.oversampling)
        self.layout = flags.layout

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        data = {"image": np.load(self.image_list[idx]), "label": np.load(self.label_list[idx])}
        if self.layout == "NDHWC":
            data["image"] = np.moveaxis(data["image"], 0, -1)
            data["label"] = np.moveaxis(data["label"], 0, -1)
        data = self.transforms(data)
        return data["image"], data["label"]
