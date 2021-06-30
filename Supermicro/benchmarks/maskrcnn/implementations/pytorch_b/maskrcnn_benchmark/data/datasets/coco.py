# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
from torchvision.io.image import ImageReadMode
import torch.multiprocessing as mp

from maskrcnn_benchmark.structures.image_list import ImageList, to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
from maskrcnn_benchmark.utils.timed_section import TimedSection

import os
import pickle
import numpy as np
import nvidia.dali as dali
from nvidia.dali.plugin.pytorch import DALIGenericIterator, feed_ndarray

min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None, pkl_ann_file=None
    ):
        if pkl_ann_file and os.path.exists(pkl_ann_file):
            with TimedSection("Unpickled COCO in %.3f seconds."):
                with open(pkl_ann_file, "rb") as f:
                    unpickled = pickle.loads(f.read())
                    self.root = root
                    self.coco = unpickled["coco"]
                    self.ids = unpickled["ids"]
                    self.transform = None
                    self.target_transform = None
                    self.transforms = None
        else:
            super(COCODataset, self).__init__(root, ann_file)
            # sort indices for reproducible results
            self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms
        self._hybrid = False

    def build_target(self, anno, img_size, pin_memory=False):
        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.tensor(boxes, dtype=torch.float32, pin_memory=pin_memory).reshape(-1, 4) # guard against no boxes
        target = BoxList(boxes, img_size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes, dtype=torch.float32, pin_memory=pin_memory)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img_size, pin_memory=pin_memory)
        target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img_size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)
        return target

    def __getitem__(self, idx):
        if self._hybrid:
            # return decoded raw image as byte tensor
            #orig_img, _ = super(COCODataset, self).__getitem__(idx)
            #orig_img_tensor = torchvision.transforms.functional.to_tensor(orig_img)
            img = torchvision.io.read_image(self.get_raw_img_info(idx), ImageReadMode.RGB)
            #print("orig_img.size = %s, img.shape = %s, orig_img_tensor.shape = %s" % (str(orig_img.size), str(img.shape), str(orig_img_tensor.shape)))
            target = self.get_target(idx)
            return img, target, idx
        else:
            img, anno = super(COCODataset, self).__getitem__(idx)
            target = self.build_target(anno, img.size)

            #orig_img, _ = super(COCODataset, self).__getitem__(idx)
            #orig_img_tensor = torchvision.transforms.functional.to_tensor(orig_img)
            #img = torchvision.io.read_image(self.get_raw_img_info(idx), ImageReadMode.RGB)
            #print("orig_img.size = %s, img.shape = %s, orig_img_tensor.shape = %s" % (str(orig_img.size), str(img.shape), str(orig_img_tensor.shape)))
            #target = self.get_target(idx)

            if self._transforms is not None:
                img, target = self._transforms(img, target)
            #print("img.shape = %s, target = %s, img.sum = %f" % (str(img.shape), str(target), img.float().sum()))
            return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

    def get_raw_img_info(self, index):
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return os.path.join(self.root, path)

    def get_target(self, index, pin_memory=False):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)
        img_size = (self.coco.imgs[img_id]["width"], self.coco.imgs[img_id]["height"])
        return self.build_target(anno, img_size, pin_memory=pin_memory)


def load_file(path):
    with open(path, 'rb') as f:
        raw_image = np.frombuffer(f.read(), dtype=np.uint8)
    return raw_image

class COCODALIBatchIterator(object):
    def __init__(self, batch_size, batch_sampler, dataset):
        self.batch_size = batch_size
        self.batch_sampler = batch_sampler
        self.batch_sampler_iter = None
        self.num_samples = len(self.batch_sampler)
        self.dataset = dataset

    def __iter__(self):
        self.batch_sampler_iter = iter(self.batch_sampler)
        return self

    def __len__(self):
        return self.num_samples

    def __next__(self):
        batch = [(load_file(self.dataset.get_raw_img_info(index)),index) for index in next(self.batch_sampler_iter)]
        raw_images, indices = tuple(zip(*batch))
        raw_images, indices = list(raw_images), list(indices)
        nshort = self.batch_size - len(raw_images)
        if nshort > 0:
            # DALI pipeline dislikes incomplete batches, so pad
            raw_images = raw_images + [raw_images[0]]*nshort
            indices = indices + [-1]*nshort
        return [raw_images, np.asarray(indices)]


class COCODALIPipeline(dali.pipeline.Pipeline):
    def __init__(self, cfg, is_train, batch_size, num_threads, device_id, seed, batch_iterator, fp16, shapes):
        super(COCODALIPipeline, self).__init__(batch_size, num_threads, device_id, seed = seed)
        self.shapes = shapes
        self.size_divisible = cfg.DATALOADER.SIZE_DIVISIBILITY
        if is_train:
            self.prob = 0.5
            self.min_size = cfg.INPUT.MIN_SIZE_TRAIN[0] if isinstance(cfg.INPUT.MIN_SIZE_TRAIN, tuple) else cfg.INPUT.MIN_SIZE_TRAIN
            self.max_size = cfg.INPUT.MAX_SIZE_TRAIN[0] if isinstance(cfg.INPUT.MAX_SIZE_TRAIN, tuple) else cfg.INPUT.MAX_SIZE_TRAIN
            self.need_targets = True
        else:
            self.prob = 0.0
            self.min_size = cfg.INPUT.MIN_SIZE_TEST[0] if isinstance(cfg.INPUT.MIN_SIZE_TEST, tuple) else cfg.INPUT.MIN_SIZE_TEST
            self.max_size = cfg.INPUT.MAX_SIZE_TEST[0] if isinstance(cfg.INPUT.MAX_SIZE_TEST, tuple) else cfg.INPUT.MAX_SIZE_TEST
            self.need_targets = False
        self.mean = torch.tensor(cfg.INPUT.PIXEL_MEAN, device='cuda').reshape([1,1,-1])
        self.stddev = torch.tensor(cfg.INPUT.PIXEL_STD, device='cuda').reshape([1,1,-1])
        self.use_gpu = cfg.DATALOADER.DALI_ON_GPU
        self.bgr = cfg.INPUT.TO_BGR255
        self.fp16 = fp16
        self.batch_iterator = batch_iterator
        self.pyt_tensor = None
        self.pyt_tensor_padded = None

    # a hack to compute sizes for resize operation in dali graph
    def if_then(self, test, a, b):
        return test * a + (1 - test) * b

    def get_min(self, a, b):
        return self.if_then(a < b, a, b)

    def get_max(self, a, b):
        return self.if_then(b < a, a, b)

    def round_down(self, a):
        round_nearest = 1.0 * dali.fn.cast(a, dtype=dali.types.INT32)
        return self.if_then(round_nearest > a, round_nearest - 1.0, round_nearest)

    def get_size(self, w, h):
        max_size = self.max_size
        min_original_size = self.get_min(w, h)
        max_original_size = self.get_max(w, h)
        size = self.if_then(max_original_size / min_original_size * self.min_size > max_size,
            max_size * min_original_size / max_original_size, self.min_size)
        size = dali.fn.cast(size, dtype=dali.types.INT32)

        test = dali.fn.cast(size, dtype=dali.types.INT32) == dali.fn.cast(min_original_size, dtype=dali.types.INT32)
        test1 = w < h
        oh = self.round_down(self.if_then(test, h, self.if_then(test1, size * h / w, size)))
        ow = self.round_down(self.if_then(test, w, self.if_then(test1, size, size * w / h)))

        return oh, ow

    def define_graph(self):
        raw_images, indices = dali.fn.external_source(source=self.batch_iterator, num_outputs=2, device="cpu")
        shapes = dali.fn.cast(dali.fn.peek_image_shape(raw_images), dtype=dali.types.FLOAT)
        h = dali.fn.slice(shapes, 0, 1, axes=[0])
        w = dali.fn.slice(shapes, 1, 1, axes=[0])
        images = dali.fn.image_decoder(raw_images, device = "mixed" if self.use_gpu else "cpu", output_type = dali.types.DALIImageType.BGR if self.bgr else dali.types.DALIImageType.RGB)
        oh, ow = self.get_size(w, h)
        images = dali.fn.resize(images, resize_x=ow, resize_y=oh)
        hori_flip = dali.fn.coin_flip(probability=self.prob)
        images = dali.fn.flip(images, horizontal=hori_flip)
        images = dali.fn.normalize(images, mean=self.mean, stddev=self.stddev)
        image_shapes = dali.fn.shapes(images)
        images = dali.fn.pad(images, align=[self.size_divisible,self.size_divisible], axis_names='HW')
        if self.fp16:
            images = dali.fn.cast(images, dtype=dali.types.DALIDataType.FLOAT16)
        else:
            images = dali.fn.cast(images, dtype=dali.types.DALIDataType.FLOAT)
        return (images, image_shapes, indices, hori_flip)

    def run(self):
        with torch.no_grad():
            pipe_out = super().share_outputs()
            super().schedule_run()
            images, image_shapes, indices, hori_flips = pipe_out
            if isinstance(image_shapes, dali.backend.TensorListGPU):
                image_shapes = image_shapes.as_cpu()
            image_shapes = image_shapes.as_array()
            image_shapes = [[img_size[0],img_size[1]] for img_size in image_shapes]
            hori_flips = hori_flips.as_array()
            indices = indices.as_array()
            outputs = []
            images = images.as_tensor()
            if self.pyt_tensor is None or self.pyt_tensor.size() != images.shape():
                self.pyt_tensor = torch.empty(size=images.shape(), dtype=torch.float16 if self.fp16 else torch.float32, device='cuda' if self.use_gpu else 'cpu')
            feed_ndarray(images, self.pyt_tensor, cuda_stream = torch.cuda.current_stream())
            if self.need_targets:
                targets = []
                for index, hori_flip, img_size in zip(indices, hori_flips, image_shapes):
                    if index >= 0:
                        target = self.batch_iterator.dataset.get_target(index)
                        if hori_flip: target = target.transpose(0)
                        target = target.resize((img_size[1],img_size[0]))
                        targets.append(target)
            else:
                targets = None
            num_raw = np.count_nonzero(indices+1)
            if self.batch_iterator.batch_size > num_raw:
                # remove padding
                self.pyt_tensor = self.pyt_tensor.narrow(0,0,num_raw)
                image_shapes = image_shapes[0:num_raw]
                indices = indices[0:num_raw]
            if self.shapes is not None:
                N, H, W, C = list(self.pyt_tensor.size())
                cost, H_best, W_best = None, None, None
                for H_pad, W_pad in self.shapes:
                    if H <= H_pad and W <= W_pad:
                        if cost is None or H_pad*W_pad < cost:
                            cost, H_best, W_best = H_pad*W_pad, H_pad, W_pad
                numel_needed = N*H_best*W_best*C
                padded_image_shape = (N, H_best, W_best, C)
                if self.pyt_tensor_padded is None or self.pyt_tensor_padded.numel() < numel_needed:
                    self.pyt_tensor_padded = self.pyt_tensor.new(size=[numel_needed])
                padded_tensor = self.pyt_tensor_padded[:numel_needed].reshape(padded_image_shape)
                padded_tensor.zero_()
                padded_tensor[:,:H,:W,:].copy_(self.pyt_tensor)
                image_list = ImageList(padded_tensor, image_shapes)
            else:
                image_list = ImageList(self.pyt_tensor, image_shapes)
            super().release_outputs()
            return image_list, targets, indices


class COCODALIDataloader(object):
    def __init__(self, cfg, is_train, device_id, batch_size, seed, batch_sampler, dataset, is_fp16, shapes):
        self.dataset = dataset
        self.batch_iterator = COCODALIBatchIterator(batch_size, batch_sampler, dataset)
        self.dali_pipeline = COCODALIPipeline(cfg, is_train, batch_size, batch_size, device_id, seed, self.batch_iterator, is_fp16, shapes)
        self.dali_pipeline.build()
        self.dali_pipeline.schedule_run()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.batch_iterator)

    def __next__(self):
        return self.dali_pipeline.run()


class HybridDataLoader(object):
    def __init__(self, cfg, is_train, batch_size, batch_sampler, dataset, collator, transforms, size_divisible, shapes):
        dataset._hybrid = True
        assert(dataset._transforms is None), "dataset.transforms must be None when hybrid dataloader is selected"
        self.dataset = dataset
        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            pin_memory=True,
        )
        self.iter = None
        self.transforms = transforms
        self.size_divisible = size_divisible
        self.shapes = shapes

    def __iter__(self):
        self.iter = iter(self.data_loader)
        return self

    def __len__(self):
        return len(self.data_loader)

    def __next__(self):
        images, targets = [], []
        raw_images, raw_targets, idxs = next(self.iter)
        for raw_image, raw_target in zip(raw_images, raw_targets):
            image = raw_image.cuda()
            image, target = self.transforms(image, raw_target)
            images.append( image )
            targets.append( target )
            #print("image.shape = %s, target = %s, image.sum = %f" % (str(image.shape), str(target), image.float().sum()))
        images = to_image_list(images, self.size_divisible, self.shapes)
        #print("images.tensors.size = %s, images.image_sizes = %s" % (str(images.tensors.size()), str(images.image_sizes)))
        return images, targets, idxs

done = mp.Event()

def hybrid_loader_worker(rank, size, batch_sampler, dataset, txbufs, q):
    j = 0
    for i, batch in enumerate(batch_sampler):
        if i % size == rank:
            metadata = []
            for idx, txbuf in zip(batch, txbufs[j]):
                img = torchvision.io.read_image(dataset.get_raw_img_info(idx), ImageReadMode.RGB)
                txbuf[:img.numel()].copy_(img.flatten())
                metadata.append( (list(img.size()), idx) )
            q.put( (j, metadata) )
            j = (j + 1) % 3
    done.wait()

class HybridDataLoader2(object):
    def __init__(self, cfg, is_train, batch_size, batch_sampler, dataset, collator, transforms, size_divisible, shapes):
        dataset._hybrid = True
        assert(dataset._transforms is None), "dataset._transforms must be None when hybrid dataloader is selected"
        self.batch_size = batch_size
        self.batch_sampler = batch_sampler
        self.dataset = dataset
        self.i = 0
        self.length = len(self.batch_sampler)
        self.transforms = transforms
        self.size_divisible = size_divisible
        self.shapes = shapes
        self.num_workers=cfg.DATALOADER.NUM_WORKERS
        maxsize = cfg.INPUT.MAX_SIZE_TRAIN if is_train else cfg.INPUT.MAX_SIZE_TEST
        self.workers, self.queues, self.txbufs = [], [], []
        for worker in range(self.num_workers):
            txbuf = [torch.empty(size=[batch_size,3*maxsize*maxsize], dtype=torch.uint8).pin_memory() for _ in range(3)]
            for t in txbuf: t.share_memory_()
            self.txbufs.append( txbuf )
            q = mp.Queue(maxsize=1)
            self.queues.append( q )
            p = mp.Process(target=hybrid_loader_worker, args=(worker,self.num_workers,batch_sampler,dataset,txbuf,q,))
            self.workers.append( p )
            p.start()

    def __del__(self):
        for p in self.workers:
            p.terminate()

    def __iter__(self):
        self.i = 0
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        if self.i < self.length:
            worker = self.i % self.num_workers
            p, q, txbufs = self.workers[worker], self.queues[worker], self.txbufs[worker]
            images, targets, idxs = [], [], []
            j, metadata = q.get()
            for txbuf, (img_size, idx) in zip(txbufs[j], metadata):
                numel = img_size[0] * img_size[1] * img_size[2]
                raw_image = txbuf[:numel].reshape(img_size)
                raw_image = raw_image.to(device='cuda', non_blocking=True)
                raw_target = self.dataset.get_target(idx, pin_memory=True)
                image, target = self.transforms(raw_image, raw_target)
                images.append( image )
                targets.append( target )
                idxs.append( idx )
            images = to_image_list(images, self.size_divisible, self.shapes)
            self.i += 1
            return images, targets, idxs
        else:
            done.set()
            raise StopIteration()

class HybridDataLoader3(object):
    def __init__(self, cfg, is_train, batch_size, batch_sampler, dataset, collator, transforms, size_divisible, shapes):
        dataset._hybrid = True
        assert(dataset._transforms is None), "dataset._transforms must be None when hybrid dataloader is selected"
        self.batch_size = batch_size
        self.length = len(batch_sampler)
        self.batch_sampler = iter(batch_sampler)
        self.dataset = dataset
        self.transforms = transforms
        self.size_divisible = size_divisible
        self.shapes = shapes

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        images, targets, idxs = [], [], []
        for idx in next(self.batch_sampler):
            raw_image = torchvision.io.read_image(self.dataset.get_raw_img_info(idx), ImageReadMode.RGB).pin_memory().to(device='cuda', non_blocking=True)
            raw_target = self.dataset.get_target(idx, pin_memory=True)
            image, target = self.transforms(raw_image, raw_target)
            images.append( image )
            targets.append( target )
            idxs.append( idx )
        images = to_image_list(images, self.size_divisible, self.shapes)
        return images, targets, idxs

