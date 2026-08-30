"""Data loaders for U-Net training (faithful, consolidated copy of the my_unet-uncertainty
2D spleen pipeline: datasets/data_loader.py + two_dim/NumpyDataLoader.py +
two_dim/NumpyDataLoader_spleen.py + two_dim/data_augmentation.py).

Reads the SAME 2-channel (image, label) npy at (2, Z, S, S). TRAIN keeps only organ-bearing
axial slices + `margin` empty neighbours (class balance); VAL/TEST score every slice.
Uses batchgenerators for the elastic augmentation. DKFZ, Apache-2.0.
"""

import fnmatch
import os
import random
from collections import defaultdict
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from batchgenerators.dataloading import SlimDataLoaderBase
from batchgenerators.transforms import Compose, MirrorTransform
from batchgenerators.transforms.spatial_transforms import ResizeTransform, SpatialTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor


# ------------------------------ augmentation ---------------------------------
def get_transforms(mode="train", target_size=128):
    tl = []
    if mode == "train":
        tl = [
            ResizeTransform(target_size=(target_size, target_size), order=1),
            MirrorTransform(axes=(1,)),
            SpatialTransform(patch_size=(target_size, target_size), random_crop=False,
                             patch_center_dist_from_border=target_size // 2,
                             do_elastic_deform=True, alpha=(0., 900.), sigma=(20., 30.),
                             do_rotation=True, p_rot_per_sample=0.8,
                             angle_x=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                             angle_y=(0, 1e-8), angle_z=(0, 1e-8),
                             scale=(0.85, 1.25), p_scale_per_sample=0.8,
                             border_mode_data="nearest", border_mode_seg="nearest"),
        ]
    elif mode in ("val", "test"):
        tl = [ResizeTransform(target_size=target_size, order=1)]
    tl.append(NumpyToTensor())
    return Compose(tl)


# ------------------------- multithreaded wrapper -----------------------------
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _worker_init(worker_id, base_seed):
    set_seed(worker_id + base_seed)


class WrappedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.transform = transform
        self.dataset = dataset
        self.is_indexable = (hasattr(dataset, "__getitem__")
                             and not (hasattr(dataset, "use_next") and dataset.use_next is True))

    def __getitem__(self, index):
        item = self.dataset[index] if self.is_indexable else next(self.dataset)
        return self.transform(**item)

    def __len__(self):
        return int(self.dataset.num_batches)


class MultiThreadedDataLoader(object):
    def __init__(self, data_loader, transform, num_processes, **kwargs):
        self.cntr = 1
        self.ds_wrapper = WrappedDataset(data_loader, transform)
        self.generator = DataLoader(self.ds_wrapper, batch_size=1, shuffle=False, sampler=None,
                                    batch_sampler=None, num_workers=num_processes,
                                    pin_memory=torch.cuda.is_available(), drop_last=False,
                                    persistent_workers=num_processes > 0,
                                    worker_init_fn=partial(_worker_init, base_seed=self.cntr))
        self.num_processes = num_processes
        self.iter = None

    def __iter__(self):
        self.iter = iter(self.generator); return self.iter

    def __next__(self):
        if self.iter is None:
            self.iter = iter(self.generator)
        return next(self.iter)

    def renew(self):
        self.cntr += 1; self.iter = iter(self.generator)

    def restart(self):
        pass


# ------------------------------ base 2D loader -------------------------------
def load_dataset(base_dir, pattern='*.npy', slice_offset=0, keys=None):
    fls, files_len, slices_ax = [], [], []
    for root, _, files in os.walk(base_dir):
        i = 0
        for filename in sorted(fnmatch.filter(files, pattern)):
            if keys is not None and filename[:-4] in keys:
                npy_file = os.path.join(root, filename)
                arr = np.load(npy_file, mmap_mode="r")
                fls.append(npy_file)
                files_len.append(arr.shape[1])
                slices_ax.extend([(i, j) for j in range(slice_offset, files_len[-1] - slice_offset)])
                i += 1
    return fls, files_len, slices_ax


class NumpyDataLoader(SlimDataLoaderBase):
    def __init__(self, base_dir, mode="train", batch_size=16, num_batches=10000000,
                 file_pattern='*.npy', label_slice=1, input_slice=(0,), keys=None):
        self.files, self.file_len, self.slices = load_dataset(base_dir, file_pattern, 0, keys)
        super().__init__(self.slices, batch_size, num_batches)
        self.batch_size = batch_size
        self.use_next = False
        self.slice_idxs = list(range(len(self.slices)))
        self.data_len = len(self.slices)
        self.num_batches = min((self.data_len // self.batch_size) + 10, num_batches)
        if isinstance(label_slice, int):
            label_slice = (label_slice,)
        self.input_slice = input_slice
        self.label_slice = label_slice
        self.np_data = np.asarray(self.slices)

    def reshuffle(self):
        random.shuffle(self.slice_idxs)

    def generate_train_batch(self):
        open_arr = random.sample(self._data, self.batch_size)
        return self.get_data_from_array(open_arr)

    def __len__(self):
        return min(self.data_len // self.batch_size, self.num_batches)

    def __getitem__(self, item):
        data_len = len(self.slices)
        start_idx = (item * self.batch_size) % data_len
        stop_idx = ((item + 1) * self.batch_size) % data_len
        if ((item + 1) * self.batch_size) == data_len:
            stop_idx = data_len
        if item > len(self) or (item * self.batch_size) == data_len or stop_idx <= start_idx:
            raise StopIteration()
        idxs = self.slice_idxs[start_idx:stop_idx]
        return self.get_data_from_array(self.np_data[idxs])

    def get_data_from_array(self, open_array):
        data, fnames, slice_idxs, labels = [], [], [], []
        for slice in open_array:
            fn_name = self.files[slice[0]]
            numpy_array = np.load(fn_name)
            numpy_slice = numpy_array[:, slice[1], ]
            data.append(numpy_slice[list(self.input_slice)])
            if self.label_slice is not None:
                labels.append(numpy_slice[list(self.label_slice)])
            fnames.append(self.files[slice[0]])
            slice_idxs.append(slice[1])
        ret = {'data': np.asarray(data), 'fnames': fnames, 'slice_idxs': slice_idxs}
        if self.label_slice is not None:
            ret['seg'] = np.asarray(labels)
        return ret


class NumpyDataSet(object):
    def __init__(self, base_dir, mode="train", batch_size=16, num_batches=10000000,
                 num_processes=8, num_cached_per_queue=8 * 4, target_size=256,
                 file_pattern='*.npy', label_slice=1, input_slice=(0,), do_reshuffle=True, keys=None):
        data_loader = NumpyDataLoader(base_dir, mode=mode, batch_size=batch_size,
                                      num_batches=num_batches, file_pattern=file_pattern,
                                      input_slice=input_slice, label_slice=label_slice, keys=keys)
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.do_reshuffle = do_reshuffle
        self.number_of_slices = 1
        self.transforms = get_transforms(mode=mode, target_size=target_size)
        self.augmenter = MultiThreadedDataLoader(data_loader, self.transforms,
                                                 num_processes=num_processes,
                                                 num_cached_per_queue=num_cached_per_queue,
                                                 shuffle=do_reshuffle)
        self.augmenter.restart()

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        if self.do_reshuffle:
            self.data_loader.reshuffle()
        self.augmenter.renew()
        return self.augmenter

    def __next__(self):
        return next(self.augmenter)


# ------------------- spleen: foreground-aware TRAIN loader --------------------
class NumpyDataLoaderSpleen(NumpyDataLoader):
    def __init__(self, base_dir, mode="train", batch_size=16, num_batches=10000000,
                 file_pattern='*.npy', label_slice=1, input_slice=(0,), keys=None,
                 foreground_only=True, margin=3, label_channel=1):
        super().__init__(base_dir, mode=mode, batch_size=batch_size, num_batches=num_batches,
                         file_pattern=file_pattern, label_slice=label_slice,
                         input_slice=input_slice, keys=keys)
        if foreground_only:
            self._keep_foreground(margin, label_channel)

    def _keep_foreground(self, margin, label_channel):
        by_file = defaultdict(list)
        for fi, sj in self.slices:
            by_file[fi].append(sj)
        kept = []
        for fi, sjs in by_file.items():
            lbl = np.asarray(np.load(self.files[fi], mmap_mode="r")[label_channel])
            fg_z = np.where((lbl > 0).any(axis=(1, 2)))[0]
            if fg_z.size == 0:
                continue
            keep_z = set()
            for z in fg_z.tolist():
                keep_z.update(range(z - margin, z + margin + 1))
            kept.extend((fi, sj) for sj in sjs if sj in keep_z)
        if not kept:
            raise RuntimeError("foreground filtering removed every slice -- check labels")
        self.slices = kept
        self._data = self.slices
        self.slice_idxs = list(range(len(self.slices)))
        self.data_len = len(self.slices)
        self.num_batches = (self.data_len // self.batch_size) + 10
        self.np_data = np.asarray(self.slices)


class NumpyDataSetSpleen(object):
    def __init__(self, base_dir, mode="train", batch_size=16, num_batches=10000000,
                 num_processes=8, num_cached_per_queue=8 * 4, target_size=256,
                 file_pattern='*.npy', label_slice=1, input_slice=(0,), do_reshuffle=True,
                 keys=None, foreground_only=True, margin=3, label_channel=1):
        data_loader = NumpyDataLoaderSpleen(base_dir, mode=mode, batch_size=batch_size,
                                            num_batches=num_batches, file_pattern=file_pattern,
                                            input_slice=input_slice, label_slice=label_slice,
                                            keys=keys, foreground_only=foreground_only,
                                            margin=margin, label_channel=label_channel)
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.do_reshuffle = do_reshuffle
        self.number_of_slices = 1
        self.transforms = get_transforms(mode=mode, target_size=target_size)
        self.augmenter = MultiThreadedDataLoader(data_loader, self.transforms,
                                                 num_processes=num_processes,
                                                 num_cached_per_queue=num_cached_per_queue,
                                                 shuffle=do_reshuffle)
        self.augmenter.restart()

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        if self.do_reshuffle:
            self.data_loader.reshuffle()
        self.augmenter.renew()
        return self.augmenter

    def __next__(self):
        return next(self.augmenter)
