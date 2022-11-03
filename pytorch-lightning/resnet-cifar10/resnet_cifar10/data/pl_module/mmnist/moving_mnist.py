"""
Fork from https://github.com/gaozhangyang/SimVP-Simpler-yet-Better-Video-Prediction/blob/master/API/dataloader_moving_mnist.py
"""

import gzip
import math
import numpy as np
import os
from PIL import Image
import random
from pytorch_lightning import LightningDataModule
import torch
import torch.utils.data as data
import ipdb

from torch.functional import F


def load_mnist(root):
    # test
    # Load MNIST dataset for generating training data.
    path = os.path.join(root, "train-images-idx3-ubyte.gz")
    with gzip.open(path, "rb") as f:
        mnist = np.frombuffer(f.read(), np.uint8, offset=16)
        mnist = mnist.reshape(-1, 28, 28)
    return mnist


def load_fixed_set(root):
    # Load the fixed dataset
    filename = "mnist_test_seq.npy"
    path = os.path.join(root, filename)
    dataset = np.load(path)
    dataset = dataset[..., np.newaxis]
    return dataset


class CustomDataModule(LightningDataModule):
    def __init__(
        self,
        data_location,
        train_batch_size,
        test_batch_size,
        in_seq_len,
        out_seq_len,
        image_size,
    ):
        super().__init__()

        self.data_location = data_location
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.image_size = image_size
        # num workers = number of cpus to use
        # get number of cpu's on this device
        num_workers = os.cpu_count()
        self.num_workers = num_workers

        # ipdb.set_trace()

        self.train_data = MovingMNIST(
            self.data_location,
            self.image_size,
            True,
            self.in_seq_len,
            self.out_seq_len,
            # transform=self.split_tranform,
        )
        self.test_data = MovingMNIST(
            self.data_location,
            self.image_size,
            False,
            self.in_seq_len,
            self.out_seq_len,
        )

    def train_dataloader(self):
        return data.DataLoader(
            self.train_data,
            batch_size=self.train_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_data,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.test_data,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class MovingMNIST(data.Dataset):
    def __init__(
        self,
        root,
        image_size,
        is_train=True,
        n_frames_input=10,
        n_frames_output=10,
        num_objects=[2],
        transform=None,
    ):
        super(MovingMNIST, self).__init__()

        self.dataset = None
        if is_train:
            self.mnist = load_mnist(root)
        else:
            if num_objects[0] != 2:
                self.mnist = load_mnist(root)
            else:
                self.dataset = load_fixed_set(root)
        self.length = int(1e4) if self.dataset is None else self.dataset.shape[1]

        self.is_train = is_train
        self.num_objects = num_objects
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform
        # For generating data
        self.image_size_ = image_size
        self.digit_size_ = 28
        self.step_length_ = 0.1
        # ipdb.set_trace()
        self.mean = 0
        self.std = 1

    def get_random_trajectory(self, seq_length):
        """Generate a random sequence of a MNIST digit"""
        canvas_size = self.image_size_ - self.digit_size_
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)
        # self.length = 500

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)
        for i in range(seq_length):
            # Take a step along velocity.
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            # Bounce off edges.
            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y
            start_y[i] = y
            start_x[i] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def generate_moving_mnist(self, num_digits=2):
        """
        Get random trajectories for the digits and generate a video.
        """
        data = np.zeros(
            (self.n_frames_total, self.image_size_, self.image_size_), dtype=np.float32
        )

        for n in range(num_digits):
            # Trajectory
            start_y, start_x = self.get_random_trajectory(self.n_frames_total)
            ind = random.randint(0, self.mnist.shape[0] - 1)
            digit_image = self.mnist[ind]
            for i in range(self.n_frames_total):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.digit_size_
                right = left + self.digit_size_
                # Draw digit
                data[i, top:bottom, left:right] = np.maximum(
                    data[i, top:bottom, left:right], digit_image
                )

        data = data[..., np.newaxis]
        return data

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output
        if self.is_train or self.num_objects[0] != 2:
            # Sample number of objects
            num_digits = random.choice(self.num_objects)
            # Generate data on the fly
            images = self.generate_moving_mnist(num_digits)
        else:
            images = self.dataset[:, idx, ...]
            # if self.image_size_ != 64:
            # Resize images
            # tensorflow resize_bilinear

        r = 1
        w = int(self.image_size_ / r)

        images = torch.from_numpy(images / 255).float()
        # print(images.shape)

        images = images.permute(0, 3, 1, 2)
        if images.shape[1] != self.image_size_:
            images = F.interpolate(
                images,
                size=(self.image_size_, self.image_size_),
                mode="bilinear",
                align_corners=True,
            )

        # print(images.shape)
        # images = images.permute(0, 3, 1, 2)

        # print(images.shape)

        input = images[: self.n_frames_input]
        output = images[
            self.n_frames_input : self.n_frames_input + self.n_frames_output
        ]
        return input, output

    def __len__(self):
        return self.length  # if self.is_train else self.length
