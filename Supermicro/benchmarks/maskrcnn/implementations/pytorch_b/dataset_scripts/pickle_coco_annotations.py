import torch
import torchvision
import pickle
import argparse

class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(self, root, ann_file):
        super(COCODataset, self).__init__(root, ann_file)

    def pickle_annotations(self, pickle_output_file):
        with open(pickle_output_file, "wb") as f:
            pickle.dump(self.coco, f)
        print("Wrote pickled annotations to %s" % (pickle_output_file))


def main(args):
    coco = COCODataset(args.root, args.ann_file)
    coco.pickle_annotations(args.pickle_output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pickle the training annotations file")
    parser.add_argument("--root", help="detectron2 dataset directory", default="/coco")
    parser.add_argument("--ann_file", help="coco training annotation file path",
                            default="/coco/annotations/instances_train2017.json")
    parser.add_argument("--pickle_output_file", help="pickled coco training annotation file output path",
                            default="/pkl_coco/instances_train2017.json.pickled")
    args=parser.parse_args()
    main(args)