from numpy.core.fromnumeric import clip, searchsorted
import torch
from torch import select
from torch.utils import data
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torch import Tensor

import numpy as np
from PIL import Image
from pathlib import Path
import os
import copy
from typing import List
from tqdm import tqdm
import random

import cv2


def get_dataloader(
    data_set_name,
    batch_size,
    data_set_dir,
    test_past_frames=10,
    test_future_frames=10,
    ngpus=1,
    num_workers=1,
    visc_list=False,
    transform_norm=False,
):
    if data_set_name == "KTH":
        norm_transform = VidNormalize(mean=0.6013795, std=2.7570653)
        renorm_transform = VidReNormalize(mean=0.6013795, std=2.7570653)
        train_transform = transforms.Compose(
            [
                VidCenterCrop((120, 120)),
                VidResize((64, 64)),
                VidRandomHorizontalFlip(0.5),
                VidRandomVerticalFlip(0.5),
                VidToTensor(),
                norm_transform,
            ]
        )
        test_transform = transforms.Compose(
            [
                VidCenterCrop((120, 120)),
                VidResize((64, 64)),
                VidToTensor(),
                norm_transform,
            ]
        )

        val_person_ids = [random.randint(1, 17)]
        KTHTrainData = KTHDataset(
            data_set_dir,
            transform=train_transform,
            train=True,
            val=True,
            num_past_frames=test_past_frames,
            num_future_frames=test_future_frames,
            val_person_ids=val_person_ids,
        )  # , actions = ['walking_no_empty'])
        train_set, val_set = KTHTrainData()
        KTHTestData = KTHDataset(
            data_set_dir,
            transform=test_transform,
            train=False,
            val=False,
            num_past_frames=test_past_frames,
            num_future_frames=test_future_frames,
        )  # , actions = ['walking_no_empty'])
        test_set = KTHTestData()

    elif data_set_name == "Navier-Stokes":
        renorm_transform = VidReNormalize(mean=0.0, std=1.0)
        train_transform = transforms.Compose(
            [VidRandomHorizontalFlip(0.5), VidRandomVerticalFlip(0.5), VidToTensor()]
        )
        test_transform = VidToTensor()

        val_ids = []
        for i in range(200):
            rand = random.randint(0, 1000)
            while rand in val_ids:
                rand = random.randint(0, 1000)
            val_ids.append(rand)
        NSTrainData = NavierStokesDataset(
            data_set_dir,
            transform=train_transform,
            train=True,
            val=True,
            num_past_frames=test_past_frames,
            num_future_frames=test_future_frames,
            val_ids=val_ids,
        )  # , actions = ['walking_no_empty'])
        train_set, val_set = NSTrainData()
        NSTestData = NavierStokesDataset(
            data_set_dir,
            transform=test_transform,
            train=False,
            val=False,
            num_past_frames=test_past_frames,
            num_future_frames=test_future_frames,
        )  # , actions = ['walking_no_empty'])
        test_set = NSTestData()

    elif data_set_name == "Navier-Stokes-BD":
        renorm_transform = VidReNormalize(mean=0.0, std=1.0)
        train_transform = transforms.Compose(
            [VidRandomHorizontalFlip(0.5), VidRandomVerticalFlip(0.5), VidToTensor()]
        )
        test_transform = VidToTensor()

        val_sample_ids = []
        for i in range(2000):
            rand = random.randint(0, 10000)
            while rand in val_sample_ids:
                rand = random.randint(0, 10000)
            val_sample_ids.append(rand)
        NSTrainData = NavierStokesBDDataset(
            data_set_dir,
            transform=train_transform,
            train=True,
            val=True,
            num_past_frames=test_past_frames,
            num_future_frames=test_future_frames,
            val_sample_ids=val_sample_ids,
        )  # , actions = ['walking_no_empty'])
        train_set, val_set = NSTrainData()
        NSTestData = NavierStokesBDDataset(
            data_set_dir,
            transform=test_transform,
            train=False,
            val=False,
            num_past_frames=test_past_frames,
            num_future_frames=test_future_frames,
        )  # , actions = ['walking_no_empty'])
        test_set = NSTestData()

    elif data_set_name == "Navier-Stokes-field-BD":
        renorm_transform = VidReNormalize(mean=0.0, std=1.0)
        # renorm_transform = None
        train_transform = transforms.Compose(
            [VidRandomHorizontalFlip(0.5), VidRandomVerticalFlip(0.5)]
        )
        test_transform = None

        N_train = 10000
        train_sample_ids = list(range(0, N_train))
        val_sample_ids = []
        for _ in range(N_train // 5):
            rand = random.randint(0, N_train)
            while rand in val_sample_ids:
                rand = random.randint(0, N_train)
            val_sample_ids.append(rand)
        for val_id in val_sample_ids:
            train_sample_ids.remove(val_id)
        test_sample_ids = list(range(N_train, int(1.2 * N_train)))

        dataset_dir = Path(data_set_dir)
        train_set = NavierStokesFieldBDDataset(
            [
                dataset_dir.joinpath(visc).joinpath(f"ns{str(id)}.npy")
                for visc in ["V100", "V1000", "V10000"]
                for id in train_sample_ids
            ],
            train_transform,
            num_past_frames=test_past_frames,
            num_future_frames=test_future_frames,
        )
        val_set = NavierStokesFieldBDDataset(
            [
                dataset_dir.joinpath(visc).joinpath(f"ns{str(id)}.npy")
                for visc in ["V100", "V1000", "V10000"]
                for id in val_sample_ids
            ],
            train_transform,
            num_past_frames=test_past_frames,
            num_future_frames=test_future_frames,
        )
        if visc_list == "total":
            test_set = NavierStokesFieldBDDataset(
                [
                    dataset_dir.joinpath(visc).joinpath(f"ns{str(id)}.npy")
                    for visc in ["V100", "V1000", "V10000"]
                    for id in test_sample_ids
                ],
                test_transform,
                num_past_frames=test_past_frames,
                num_future_frames=test_future_frames,
            )
        else:
            test_set = NavierStokesFieldBDDataset(
                [
                    dataset_dir.joinpath(visc).joinpath(f"ns{str(id)}.npy")
                    for visc in visc_list
                    for id in test_sample_ids
                ],
                test_transform,
                num_past_frames=test_past_frames,
                num_future_frames=test_future_frames,
            )

    elif data_set_name == "Navier-Stokes-total-BD":
        # channels: data_w, data_ux, data_uy, data_uxy, data_f, data_w0, data_v
        # mean = (
        #     -2.59000000e-03,
        #     -1.45043333e-04,
        #     4.94933333e-05,
        #     -7.93300000e-05,
        #     -1.19333333e-04,
        #     -2.59000000e-03,
        #     -3.00000000e00,
        # )
        # std = (
        #     1.39928314,
        #     6.04926993,
        #     6.05185922,
        #     87.36488616,
        #     0.09993324,
        #     1.39928314,
        #     0.81649658,
        # )
        # min = (-7.45e00, -1.21e02, -1.26e02, -5.68e03, -1.41e-01, -7.45e00, -4.00e00)
        # max = (7.66e00, 1.20e02, 1.15e02, 5.59e03, 1.41e-01, 7.66e00, -2.00e00)
        # channels: data_w, data_f, data_w0, data_ux, data_uy, data_uxy, data_v
        mean = (
            -2.59000000e-03,
            -1.19333333e-04,
            -2.59000000e-03,
            -1.45043333e-04,
            4.94933333e-05,
            -7.93300000e-05,
            -3.00000000e00,
        )
        std = (
            1.39928314,
            0.09993324,
            1.39928314,
            6.04926993,
            6.05185922,
            87.36488616,
            0.81649658,
        )
        min = (-7.45e00, -1.41e-01, -7.45e00, -1.21e02, -1.26e02, -5.68e03, -4.00e00)
        max = (7.66e00, 1.41e-01, 7.66e00, 1.20e02, 1.15e02, 5.59e03, -2.00e00)
        if transform_norm == "mean_std":
            norm_transform = VidNormalize(mean=mean, std=std)
            train_transform = transforms.Compose(
                [
                    VidRandomHorizontalFlip(0.5),
                    VidRandomVerticalFlip(0.5),
                    norm_transform,
                ]
            )
            test_transform = transforms.Compose([norm_transform])
            renorm_transform = transforms.Compose(
                [VidReNormalize(mean=mean, std=std), VidMinMax(min=min, max=max)]
            )
        elif transform_norm == "min_max":
            norm_transform = VidMinMax(min=min, max=max)
            train_transform = transforms.Compose(
                [
                    VidRandomHorizontalFlip(0.5),
                    VidRandomVerticalFlip(0.5),
                    norm_transform,
                ]
            )
            test_transform = transforms.Compose([norm_transform])
            renorm_transform = None
        else:
            norm_transform = None
            train_transform = transforms.Compose(
                [
                    VidRandomHorizontalFlip(0.5),
                    VidRandomVerticalFlip(0.5),
                ]
            )
            test_transform = None
            renorm_transform = VidMinMax(min=min, max=max)

        N_train = 10000
        train_sample_ids = list(range(0, N_train))
        val_sample_ids = []
        for _ in range(N_train // 5):
            rand = random.randint(0, N_train - 1)
            while rand in val_sample_ids:
                rand = random.randint(0, N_train - 1)
            val_sample_ids.append(rand)
        for val_id in val_sample_ids:
            train_sample_ids.remove(val_id)
        test_sample_ids = list(range(N_train, int(1.2 * N_train)))

        dataset_dir = Path(data_set_dir)
        train_set = NavierStokesTotalBDDataset(
            [
                dataset_dir.joinpath(visc).joinpath(f"ns{str(id)}.npy")
                for visc in ["V100", "V1000", "V10000"]
                for id in train_sample_ids
            ],
            train_transform,
            num_past_frames=test_past_frames,
            num_future_frames=test_future_frames,
        )
        val_set = NavierStokesTotalBDDataset(
            [
                dataset_dir.joinpath(visc).joinpath(f"ns{str(id)}.npy")
                for visc in ["V100", "V1000", "V10000"]
                for id in val_sample_ids
            ],
            train_transform,
            num_past_frames=test_past_frames,
            num_future_frames=test_future_frames,
        )
        if visc_list == "total":
            test_set = NavierStokesTotalBDDataset(
                [
                    dataset_dir.joinpath(visc).joinpath(f"ns{str(id)}.npy")
                    for visc in ["V100", "V1000", "V10000"]
                    for id in test_sample_ids
                ],
                test_transform,
                num_past_frames=test_past_frames,
                num_future_frames=test_future_frames,
            )
        else:
            test_set = NavierStokesTotalBDDataset(
                [
                    dataset_dir.joinpath(visc).joinpath(f"ns{str(id)}.npy")
                    for visc in visc_list
                    for id in test_sample_ids
                ],
                test_transform,
                num_past_frames=test_past_frames,
                num_future_frames=test_future_frames,
            )

    elif data_set_name == "MNIST":
        renorm_transform = VidReNormalize(mean=0.0, std=1.0)
        train_transform = transforms.Compose(
            [VidRandomHorizontalFlip(0.5), VidRandomVerticalFlip(0.5), VidToTensor()]
        )
        test_transform = VidToTensor()

        dataset_dir = Path(data_set_dir)
        train_set = MovingMNISTDataset(
            dataset_dir.joinpath("moving-mnist-train.npz"), train_transform
        )
        val_set = MovingMNISTDataset(
            dataset_dir.joinpath("moving-mnist-valid.npz"), train_transform
        )
        test_set = MovingMNISTDataset(
            dataset_dir.joinpath("moving-mnist-test.npz"), test_transform
        )

    elif data_set_name == "BAIR":
        dataset_dir = Path(data_set_dir)
        norm_transform = VidNormalize(
            (0.61749697, 0.6050092, 0.52180636), (2.1824553, 2.1553133, 1.9115673)
        )
        renorm_transform = VidReNormalize(
            (0.61749697, 0.6050092, 0.52180636), (2.1824553, 2.1553133, 1.9115673)
        )
        # norm_transform = VidNormalize((0.6175636, 0.60508573, 0.52188003), (2.8584306, 2.8212209, 2.499153))
        # renorm_transform = VidReNormalize((0.6175636, 0.60508573, 0.52188003), (2.8584306, 2.8212209, 2.499153))
        transform = transforms.Compose([VidToTensor(), norm_transform])

        BAIR_train_whole_set = BAIRDataset(
            dataset_dir.joinpath("train"),
            transform,
            color_mode="RGB",
            num_past_frames=2,
            num_future_frames=10,
        )()
        train_val_ratio = 0.95
        BAIR_train_set_length = int(len(BAIR_train_whole_set) * train_val_ratio)
        BAIR_val_set_length = len(BAIR_train_whole_set) - BAIR_train_set_length
        train_set, val_set = random_split(
            BAIR_train_whole_set,
            [BAIR_train_set_length, BAIR_val_set_length],
            generator=torch.Generator().manual_seed(2021),
        )

        test_set = BAIRDataset(
            dataset_dir.joinpath("test"),
            transform,
            color_mode="RGB",
            num_past_frames=2,
            num_future_frames=test_future_frames,
        )()

    N = batch_size
    train_loader = DataLoader(
        train_set, batch_size=N, shuffle=True, num_workers=num_workers, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=N, shuffle=True, num_workers=num_workers, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=N, shuffle=True, num_workers=num_workers, drop_last=False
    )

    if ngpus > 1:
        N = batch_size // ngpus
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)

        train_loader = DataLoader(
            train_set,
            batch_size=N,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            sampler=train_sampler,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=N,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            sampler=val_sampler,
            drop_last=True,
        )

    return train_loader, val_loader, test_loader, renorm_transform


class KTHDataset(object):
    """
    KTH dataset, a wrapper for ClipDataset
    the original frame size is (H, W) = (120, 160)
    Split the KTH dataset and return the train and test dataset
    """

    def __init__(
        self,
        KTH_dir,
        transform,
        train,
        val,
        num_past_frames,
        num_future_frames,
        actions=[
            "boxing",
            "handclapping",
            "handwaving",
            "jogging",
        ],
        val_person_ids=None,
    ):
        """
        Args:
            KTH_dir --- Directory for extracted KTH video frames
            train --- True for training dataset, False for test dataset
            transform --- trochvison transform functions
            num_past_frames --- number of past frames
            num_future_frames --- number of future frames
        """
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.clip_length = num_past_frames + num_future_frames
        self.transform = transform
        self.color_mode = "grey_scale"

        self.KTH_path = Path(KTH_dir).absolute()
        self.actions = actions
        self.train = train
        self.val = val
        if self.train:
            self.person_ids = list(range(1, 17))
            if self.val:
                if val_person_ids is None:  # one person for the validation
                    self.val_person_ids = [random.randint(1, 17)]
                    self.person_ids.remove(self.val_person_ids[0])
                else:
                    self.val_person_ids = val_person_ids
        else:
            self.person_ids = list(range(17, 26))

        frame_folders = self.__getFramesFolder__(self.person_ids)
        self.clips = self.__getClips__(frame_folders)

        if self.val:
            val_frame_folders = self.__getFramesFolder__(self.val_person_ids)
            self.val_clips = self.__getClips__(val_frame_folders)

    def __call__(self):
        """
        Returns:
            clip_set --- ClipDataset object
        """

        clip_set = ClipDataset(
            self.num_past_frames,
            self.num_future_frames,
            self.clips,
            self.transform,
            self.color_mode,
        )
        if self.val:
            val_clip_set = ClipDataset(
                self.num_past_frames,
                self.num_future_frames,
                self.val_clips,
                self.transform,
                self.color_mode,
            )
            return clip_set, val_clip_set
        else:
            return clip_set

    def __getClips__(self, frame_folders):
        clips = []
        for folder in frame_folders:
            img_files = sorted(list(folder.glob("*")))
            clip_num = len(img_files) // self.clip_length
            rem_num = len(img_files) % self.clip_length
            img_files = img_files[
                rem_num // 2 : rem_num // 2 + clip_num * self.clip_length
            ]
            for i in range(clip_num):
                clips.append(
                    img_files[i * self.clip_length : (i + 1) * self.clip_length]
                )

        return clips

    def __getFramesFolder__(self, person_ids):
        """
        Get the KTH frames folders for ClipDataset
        Returns:
            return_folders --- ther returned video frames folders
        """

        frame_folders = []
        for a in self.actions:
            action_path = self.KTH_path.joinpath(a)
            # print(action_path)
            frame_folders.extend(
                [
                    action_path.joinpath(s)
                    for s in os.listdir(action_path)
                    if ".avi" not in s
                ]
            )
        frame_folders = sorted(frame_folders)

        return_folders = []
        for ff in frame_folders:
            person_id = int(str(ff.name).strip().split("_")[0][-2:])
            if person_id in person_ids:
                return_folders.append(ff)

        return return_folders


class NavierStokesDataset(object):
    def __init__(
        self,
        NS_dir,
        transform,
        train,
        val,
        num_past_frames,
        num_future_frames,
        val_ids=None,
    ):

        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.clip_length = num_past_frames + num_future_frames
        self.transform = transform
        self.color_mode = "grey_scale"

        self.NS_path = Path(NS_dir).absolute()

        self.train = train
        self.val = val
        if self.train:
            self.ids = list(range(0, 1000))
            if self.val:
                if val_ids is None:
                    val_ids = []
                    for i in range(200):
                        rand = random.randint(0, 1000)
                        while rand in val_ids:
                            rand = random.randint(0, 1000)
                        val_ids.append(rand)
                    self.val_ids = val_ids
                    for vid in self.val_ids:
                        self.ids.remove(vid)
                else:
                    self.val_ids = val_ids
        else:
            self.ids = list(range(1000, 1200))

        frame_folders = self.__getFramesFolder__(self.ids)
        self.clips = self.__getClips__(frame_folders)

        if self.val:
            val_frame_folders = self.__getFramesFolder__(self.val_ids)
            self.val_clips = self.__getClips__(val_frame_folders)

    def __call__(self):
        """
        Returns:
            clip_set --- ClipDataset object
        """

        clip_set = ClipDataset(
            self.num_past_frames,
            self.num_future_frames,
            self.clips,
            self.transform,
            self.color_mode,
        )
        if self.val:
            val_clip_set = ClipDataset(
                self.num_past_frames,
                self.num_future_frames,
                self.val_clips,
                self.transform,
                self.color_mode,
            )
            return clip_set, val_clip_set
        else:
            return clip_set

    def __getClips__(self, frame_folders):
        clips = []
        for folder in frame_folders:
            img_files = sorted(list(folder.glob("*")))
            clip_num = len(img_files) // self.clip_length
            rem_num = len(img_files) % self.clip_length
            img_files = img_files[
                rem_num // 2 : rem_num // 2 + clip_num * self.clip_length
            ]
            for i in range(clip_num):
                clips.append(
                    img_files[i * self.clip_length : (i + 1) * self.clip_length]
                )

        return clips

    def __getFramesFolder__(self, ids):
        """
        Get the KTH frames folders for ClipDataset
        Returns:
            return_folders --- ther returned video frames folders
        """

        frame_folders = []

        frame_folders.extend(
            [
                self.NS_path.joinpath(s)
                for s in os.listdir(self.NS_path)
                if ".avi" not in s
            ]
        )
        frame_folders = sorted(frame_folders)

        return_folders = []
        for ff in frame_folders:
            id = int(str(ff.name).strip().split("s")[1])
            if id in ids:
                return_folders.append(ff)

        return return_folders


class NavierStokesBDDataset(object):
    def __init__(
        self,
        NS_dir,
        transform,
        train,
        val,
        num_past_frames,
        num_future_frames,
        viscs=[
            "V100",
            "V1000",
            "V10000",
        ],
        val_sample_ids=None,
    ):

        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.clip_length = num_past_frames + num_future_frames
        self.transform = transform
        self.color_mode = "grey_scale"

        self.NS_path = Path(NS_dir).absolute()
        self.viscs = viscs
        self.train = train
        self.val = val
        if self.train:
            self.sample_ids = list(range(0, 10000))
            if self.val:
                if val_sample_ids is None:  # one sample for the validation
                    val_sample_ids = []
                    for i in range(2000):
                        rand = random.randint(0, 10000)
                        while rand in val_sample_ids:
                            rand = random.randint(0, 10000)
                        val_sample_ids.append(rand)
                    self.val_sample_ids = val_sample_ids
                    for val_id in self.val_sample_ids:
                        self.sample_ids.remove(val_id)
                else:
                    self.val_sample_ids = val_sample_ids
        else:
            self.sample_ids = list(range(10000, 12000))

        frame_folders = self.__getFramesFolder__(self.sample_ids)
        self.clips = self.__getClips__(frame_folders)

        if self.val:
            val_frame_folders = self.__getFramesFolder__(self.val_sample_ids)
            self.val_clips = self.__getClips__(val_frame_folders)

    def __call__(self):
        """
        Returns:
            clip_set --- ClipDataset object
        """

        clip_set = ClipDataset(
            self.num_past_frames,
            self.num_future_frames,
            self.clips,
            self.transform,
            self.color_mode,
        )
        if self.val:
            val_clip_set = ClipDataset(
                self.num_past_frames,
                self.num_future_frames,
                self.val_clips,
                self.transform,
                self.color_mode,
            )
            return clip_set, val_clip_set
        else:
            return clip_set

    def __getClips__(self, frame_folders):
        clips = []
        for folder in frame_folders:
            img_files = sorted(list(folder.glob("*")))
            clip_num = len(img_files) // self.clip_length
            rem_num = len(img_files) % self.clip_length
            img_files = img_files[
                rem_num // 2 : rem_num // 2 + clip_num * self.clip_length
            ]
            for i in range(clip_num):
                clips.append(
                    img_files[i * self.clip_length : (i + 1) * self.clip_length]
                )

        return clips

    def __getFramesFolder__(self, sample_ids):
        """
        Get the KTH frames folders for ClipDataset
        Returns:
            return_folders --- ther returned video frames folders
        """

        frame_folders = []
        for a in self.viscs:
            visc_path = self.NS_path.joinpath(a)
            # print(visc_path)
            frame_folders.extend(
                [
                    visc_path.joinpath(s)
                    for s in os.listdir(visc_path)
                    if ".avi" not in s
                ]
            )
        frame_folders = sorted(frame_folders)

        return_folders = []
        for ff in frame_folders:
            sample_id = int(str(ff.name).strip().split("s")[1])
            if sample_id in sample_ids:
                return_folders.append(ff)

        return return_folders


class NavierStokesFieldBDDataset(object):
    def __init__(
        self, data_path_list, transform, num_past_frames=10, num_future_frames=10
    ):
        """
        both num_past_frames and num_future_frames are limited to be 10
        Args:
            data_path --- npz file path
            transfrom --- torchvision transforms for the image
        Return batched Sample:
            past_clip --- Tensor with shape (batch_size, num_past_frames, C, H, W)
            future_clip --- Tensor with shape (batch_size, num_future_frames, C, H, W)
        """
        self.data_path_list = data_path_list
        # self.data = self.load_data()

        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.transform = transform
        # print(self.num_past_frames, self.num_future_frames)

    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, index: int):
        if torch.is_tensor(index):
            index = index.to_list()

        minV = -6.5
        maxV = 6.5

        file_path = self.data_path_list[index]
        data = torch.Tensor(np.load(file_path))  # t,c,x,y
        data = (data - minV) / (maxV - minV) * 1.0
        data = torch.clamp(data, min=0.0, max=1.0)
        if self.transform is not None:
            data = self.transform(data)
        # print(data.shape)

        # print(torch.min(data), torch.max(data))

        past_clip = data[0 : self.num_past_frames, :, :, :]
        future_clip = data[
            self.num_past_frames : self.num_past_frames + self.num_future_frames,
            :,
            :,
            :,
        ]

        # print("past, future clip: ", past_clip.shape, future_clip.shape)

        return past_clip, future_clip


class NavierStokesTotalBDDataset(object):
    def __init__(
        self, data_path_list, transform, num_past_frames=10, num_future_frames=10
    ):
        """
        both num_past_frames and num_future_frames are limited to be 10
        Args:
            data_path --- npz file path
            transfrom --- torchvision transforms for the image
        Return batched Sample:
            past_clip --- Tensor with shape (batch_size, num_past_frames, C, H, W)
            future_clip --- Tensor with shape (batch_size, num_future_frames, C, H, W)
        """
        self.data_path_list = data_path_list
        # self.data = self.load_data()

        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.transform = transform
        # print(self.num_past_frames, self.num_future_frames)

    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, index: int):
        if torch.is_tensor(index):
            index = index.to_list()
        # data_w, data_f, data_w0, data_ux, data_uy, data_uxy, data_v
        # minV = -6.5
        # maxV = 6.5

        file_path = self.data_path_list[index]
        data = torch.Tensor(np.load(file_path))  # t,c,x,y
        data[:, 6, ...] = torch.log10(data[:, 6, ...])  # log viscosity
        # data = data[:, [0, 3, 4, 5, 1, 2, 6], ...]

        T, C, X, Y = data.shape
        # minV = torch.Tensor([-7.5e00, -1.5e-01, -7.5e00, -1.5e02, -1.5e02, -5.0e03, -4.0])
        # maxV = torch.Tensor([7.5e00, 1.5e-01, 7.5e00, 1.5e02, 1.5e02, 5.0e03, -2.0])
        # minV = torch.Tensor([-7.45e00, -1.41e-01, -1.05e00, -1.21e02, -1.26e02, -5.68e03, -4.00e00])
        # maxV = torch.Tensor([7.66e00, 1.41e-01, 1.02e00, 1.20e02, 1.15e02, 5.59e03, -2.00e00])
        # minV = minV.expand(T, X, Y, C).permute(0, 3, 1, 2)
        # maxV = maxV.expand(T, X, Y, C).permute(0, 3, 1, 2)
        # data = (data - minV) / (maxV - minV) * 1.0

        # data = torch.clamp(data, min=0.0, max=1.0)
        if self.transform is not None:
            data = self.transform(data)
        # print(data.shape)

        # print(torch.min(data), torch.max(data))

        past_clip = data[0 : self.num_past_frames, :, :, :]
        future_clip = data[
            self.num_past_frames : self.num_past_frames + self.num_future_frames,
            :,
            :,
            :,
        ]

        # print("past, future clip: ", past_clip.shape, future_clip.shape)
        # print(
        #     torch.concat((past_clip, future_clip), dim=0).min(),
        #     torch.concat((past_clip, future_clip), dim=0).max(),
        # )
        return past_clip, future_clip


class BAIRDataset(object):
    """
    BAIR dataset, a wrapper for ClipDataset
    the original frame size is (H, W) = (64, 64)
    The train and test frames has been previously splitted: ref "Self-Supervised Visual Planning with Temporal Skip Connections"
    """

    def __init__(
        self,
        frames_dir: str,
        transform,
        color_mode="RGB",
        num_past_frames=10,
        num_future_frames=10,
    ):
        """
        Args:
            frames_dir --- Directory of extracted video frames and original videos.
            transform --- trochvison transform functions
            color_mode --- 'RGB' or 'grey_scale' color mode for the dataset
            num_past_frames --- number of past frames
            num_future_frames --- number of future frames
            clip_length --- number of frames for each video clip example for model
        """
        self.frames_path = Path(frames_dir).absolute()
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.clip_length = num_past_frames + num_future_frames
        self.transform = transform
        self.color_mode = color_mode

        self.clips = self.__getClips__()

    def __call__(self):
        """
        Returns:
            data_set --- ClipDataset object
        """
        data_set = ClipDataset(
            self.num_past_frames,
            self.num_future_frames,
            self.clips,
            self.transform,
            self.color_mode,
        )

        return data_set

    def __getClips__(self):
        clips = []
        frames_folders = os.listdir(self.frames_path)
        frames_folders = [self.frames_path.joinpath(s) for s in frames_folders]
        for folder in frames_folders:
            img_files = sorted(list(folder.glob("*")))
            clip_num = len(img_files) // self.clip_length
            rem_num = len(img_files) % self.clip_length
            img_files = img_files[
                rem_num // 2 : rem_num // 2 + clip_num * self.clip_length
            ]
            for i in range(clip_num):
                clips.append(
                    img_files[i * self.clip_length : (i + 1) * self.clip_length]
                )

        return clips


class ClipDataset(Dataset):
    """
    Video clips dataset
    """

    def __init__(
        self, num_past_frames, num_future_frames, clips, transform, color_mode
    ):
        """
        Args:
            num_past_frames --- number of past frames
            num_future_frames --- number of future frames
            clips --- List of video clips frames file path
            transfrom --- torchvision transforms for the image
            color_mode --- 'RGB' for RGB dataset, 'grey_scale' for grey_scale dataset
        Return batched Sample:
            past_clip --- Tensor with shape (batch_size, num_past_frames, C, H, W)
            future_clip --- Tensor with shape (batch_size, num_future_frames, C, H, W)
        """
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.clips = clips
        self.transform = transform
        if color_mode != "RGB" and color_mode != "grey_scale" and color_mode != "field":
            raise ValueError("Unsupported color mode!!")
        else:
            self.color_mode = color_mode

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index: int):
        """
        Returns:
            past_clip: Tensor with shape (num_past_frames, C, H, W)
            future_clip: Tensor with shape (num_future_frames, C, H, W)
        """
        if torch.is_tensor(index):
            index = index.to_list()

        clip_imgs = self.clips[index]
        imgs = []
        for img_path in clip_imgs:
            if self.color_mode == "RGB":
                img = Image.open(img_path.absolute().as_posix()).convert("RGB")
            elif self.color_mode == "grey_scale":
                img = Image.open(img_path.absolute().as_posix()).convert("L")
            imgs.append(img)

        original_clip = self.transform(imgs)
        past_clip = original_clip[0 : self.num_past_frames, ...]
        future_clip = original_clip[-self.num_future_frames :, ...]
        return past_clip, future_clip

    def visualize_clip(self, clip, file_name):
        """
        save a video clip to GIF file
        Args:
            clip: tensor with shape (clip_length, C, H, W)
        """
        imgs = []
        for i in range(clip.shape[0]):
            img = transforms.ToPILImage()(clip[i, ...])
            imgs.append(img)

        videodims = img.size
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        video = cv2.VideoWriter(
            Path(file_name).absolute().as_posix(), fourcc, 10, videodims
        )
        for img in imgs:
            video.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        video.release()
        # imgs[0].save(str(Path(file_name).absolute()), save_all = True, append_images = imgs[1:])


class MovingMNISTDataset(Dataset):
    """
    MovingMNIST dataset
    """

    def __init__(self, data_path, transform):
        """
        both num_past_frames and num_future_frames are limited to be 10
        Args:
            data_path --- npz file path
            transfrom --- torchvision transforms for the image
        Return batched Sample:
            past_clip --- Tensor with shape (batch_size, num_past_frames, C, H, W)
            future_clip --- Tensor with shape (batch_size, num_future_frames, C, H, W)
        """
        self.data_path = data_path
        self.data = self.load_data()

        self.transform = transform

    def load_data(self):
        data = {}
        np_arr = np.load(self.data_path.absolute().as_posix())

        for key in np_arr:
            data[key] = np_arr[key]
        # print("data keys: ", data.keys())
        # print("clips: ", data["clips"].shape, data["clips"][:, 0, :])
        # print("dims: ", data["dims"].shape, data["dims"][0])
        # print("input_raw_data: ", data["input_raw_data"].shape, data["input_raw_data"][0])

        return data

    def __len__(self):
        return self.data["clips"].shape[1]

    def __getitem__(self, index: int):
        """
        Returns:
            past_clip: Tensor with shape (num_past_frames, C, H, W)
            future_clip: Tensor with shape (num_future_frames, C, H, W)
        """
        if torch.is_tensor(index):
            index = index.to_list()

        clip_index = self.data["clips"][:, index, :]
        psi, pei = clip_index[0, 0], clip_index[0, 0] + clip_index[0, 1]
        past_clip = self.data["input_raw_data"][psi:pei, ...]
        fsi, fei = clip_index[1, 0], clip_index[1, 0] + clip_index[1, 1]
        future_clip = self.data["input_raw_data"][fsi:fei, ...]

        full_clip = torch.from_numpy(np.concatenate((past_clip, future_clip), axis=0))
        imgs = []
        for i in range(full_clip.shape[0]):
            # img = full_clip[i, ...]
            # print(i, img.shape)
            img = transforms.ToPILImage()(full_clip[i, ...])
            imgs.append(img)

        full_clip = self.transform(imgs)
        past_clip = full_clip[0 : clip_index[0, 1], ...]
        future_clip = full_clip[-clip_index[1, 1] :, ...]

        return past_clip, future_clip

    def visualize_clip(self, clip, file_name):
        """
        save a video clip to GIF file
        Args:
            clip: tensor with shape (clip_length, C, H, W)
        """
        imgs = []
        for i in range(clip.shape[0]):
            img = transforms.ToPILImage()(clip[i, ...])
            imgs.append(img)

        imgs[0].save(
            str(Path(file_name).absolute()), save_all=True, append_images=imgs[1:]
        )


class VidResize(object):
    def __init__(self, *args, **resize_kwargs):
        self.resize_kwargs = resize_kwargs
        self.args = args

    def __call__(self, clip: List[Image.Image]):
        for i in range(len(clip)):
            clip[i] = transforms.Resize(*self.args, **self.resize_kwargs)(clip[i])

        return clip


class VidCenterCrop(object):
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args

    def __call__(self, clip: List[Image.Image]):
        for i in range(len(clip)):
            clip[i] = transforms.CenterCrop(*self.args, **self.kwargs)(clip[i])

        return clip


class VidCrop(object):
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args

    def __call__(self, clip: List[Image.Image]):
        for i in range(len(clip)):
            clip[i] = transforms.functional.crop(clip[i], *self.args, **self.kwargs)

        return clip


class VidRandomHorizontalFlip(object):
    def __init__(self, p: float):
        assert p >= 0 and p <= 1, "invalid flip probability"
        self.p = p

    def __call__(self, clip):
        if np.random.rand() < self.p:
            for i in range(len(clip)):
                clip[i] = transforms.functional.hflip(clip[i])
        return clip


class VidRandomVerticalFlip(object):
    def __init__(self, p: float):
        assert p >= 0 and p <= 1, "invalid flip probability"
        self.p = p

    def __call__(self, clip):
        if np.random.rand() < self.p:
            for i in range(len(clip)):
                clip[i] = transforms.functional.vflip(clip[i])
        return clip


class VidToTensor(object):
    def __call__(self, clip):
        """
        Return: clip --- Tensor with shape (T, C, H, W)
        """
        for i in range(len(clip)):
            clip[i] = transforms.ToTensor()(clip[i])
        clip = torch.stack(clip, dim=0)

        return clip


class VidNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, clip: Tensor):
        """
        Return: clip --- Tensor with shape (T, C, H, W)
        """
        T, _, _, _ = clip.shape
        for i in range(T):
            clip[i, ...] = transforms.Normalize(self.mean, self.std)(clip[i, ...])

        return clip


class VidReNormalize(object):
    def __init__(self, mean, std):
        try:
            self.inv_std = [1.0 / s for s in std]
            self.inv_mean = [-m for m in mean]
            self.renorm = transforms.Compose(
                [
                    transforms.Normalize(mean=np.zeros(len(mean)), std=self.inv_std),
                    transforms.Normalize(mean=self.inv_mean, std=np.ones(len(std))),
                ]
            )
        except TypeError:
            # try normalize for grey_scale images.
            self.inv_std = 1.0 / std
            self.inv_mean = -mean
            self.renorm = transforms.Compose(
                [
                    transforms.Normalize(mean=0.0, std=self.inv_std),
                    transforms.Normalize(mean=self.inv_mean, std=1.0),
                ]
            )

    def __call__(self, clip: Tensor):
        """
        Return: clip --- Tensor with shape (T, C, H, W)
        """
        T, _, _, _ = clip.shape
        for i in range(T):
            clip[i, ...] = self.renorm(clip[i, ...])

        return clip


class VidMinMax(object):
    def __init__(self, min, max):
        self.min = torch.Tensor(min)
        self.max = torch.Tensor(max)

    def __call__(self, clip: Tensor):
        """
        Return: clip --- Tensor with shape (T, C, H, W)
        """
        T, C, H, W = clip.shape

        for i in range(T):
            clip[i, ...] = (
                clip[i, ...] - self.min.expand(H, W, C).permute(2, 0, 1)
            ) / (
                self.max.expand(H, W, C).permute(2, 0, 1)
                - self.min.expand(H, W, C).permute(2, 0, 1)
            )

        return clip


class VidPad(object):
    """
    If pad, Do not forget to pass the mask to the transformer encoder.
    """

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args

    def __call__(self, clip: List[Image.Image]):
        for i in range(len(clip)):
            clip[i] = transforms.Pad(*self.args, **self.kwargs)(clip[i])

        return clip


def mean_std_compute(dataset, device, color_mode="RGB"):
    """
    arguments:
        dataset: pytorch dataloader
        device: torch.device('cuda:0') or torch.device('cpu') for computation
    return:
        mean and std of each image channel.
        std = sqrt(E(x^2) - (E(X))^2)
    """
    data_iter = iter(dataset)
    sum_img = None
    square_sum_img = None
    N = 0

    pgbar = tqdm(desc="summarizing...", total=len(dataset))
    for idx, sample in enumerate(data_iter):
        past, future = sample
        clip = torch.cat([past, future], dim=0)
        N += clip.shape[0]

        img = torch.sum(clip, axis=0)

        if idx == 0:
            sum_img = img
            square_sum_img = torch.square(img)
            sum_img = sum_img.to(torch.device(device))
            square_sum_img = square_sum_img.to(torch.device(device))
        else:
            img = img.to(device)
            sum_img = sum_img + img
            square_sum_img = square_sum_img + torch.square(img)

        pgbar.update(1)

    pgbar.close()

    mean_img = sum_img / N
    mean_square_img = square_sum_img / N
    if color_mode == "RGB":
        mean_r, mean_g, mean_b = (
            torch.mean(mean_img[0, :, :]),
            torch.mean(mean_img[1, :, :]),
            torch.mean(mean_img[2, :, :]),
        )
        mean_r2, mean_g2, mean_b2 = (
            torch.mean(mean_square_img[0, :, :]),
            torch.mean(mean_square_img[1, :, :]),
            torch.mean(mean_square_img[2, :, :]),
        )
        std_r, std_g, std_b = (
            torch.sqrt(mean_r2 - torch.square(mean_r)),
            torch.sqrt(mean_g2 - torch.square(mean_g)),
            torch.sqrt(mean_b2 - torch.square(mean_b)),
        )

        return (
            [mean_r.cpu().numpy(), mean_g.data.cpu().numpy(), mean_b.cpu().numpy()],
            [std_r.cpu().numpy(), std_g.cpu().numpy(), std_b.cpu().numpy()],
        )
    else:
        mean = torch.mean(mean_img)
        mean_2 = torch.mean(mean_square_img)
        std = torch.sqrt(mean_2 - torch.square(mean))

        return (mean.cpu().numpy(), std.cpu().numpy())
