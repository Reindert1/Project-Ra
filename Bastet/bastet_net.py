#!usr/bin/env python3

"""
Patch-wise CNN implementation for image segmentation
"""

__author__ = "Skippybal"
__version__ = "1.0"

import math
import os
import sys
import argparse as ap
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageSet(Dataset):
    """
    Dataset to train neural network
    """
    def __init__(self, dataloc, labels, window_size):
        self.pad = window_size
        self.rad = window_size // 2
        transform = transforms.Grayscale()
        convert_tensor = transforms.ToTensor()
        self.image = convert_tensor(transform(Image.open(dataloc)))

        self.np_labels = np.asarray(Image.open(labels)).flatten()
        self.valid_pos = np.where(self.np_labels != 0)[0]

        self.labels = torch.tensor(self.np_labels, dtype=torch.long)

        padding = transforms.Pad(window_size)

        self.padded_image = padding(self.image)

        self.shape = self.image.shape

        self.length = self.image.flatten().shape

    def __getitem__(self, index):
        index = self.valid_pos[index]

        x_position = index // self.shape[2] + self.pad
        y_position = index % self.shape[2] + self.pad

        patch = self.padded_image[:, x_position - self.rad:x_position + self.rad + 1,
                y_position - self.rad:y_position + self.rad + 1]

        return patch, self.labels[index]

    def __len__(self):
        return self.valid_pos.shape[0]


class FullClassSet(Dataset):
    """
    Dataset to segment image
    """
    def __init__(self, dataloc, window_size):
        self.pad = window_size
        self.rad = window_size // 2
        transform = transforms.Grayscale()
        convert_tensor = transforms.ToTensor()
        self.image = convert_tensor(transform(Image.open(dataloc)))

        padding = transforms.Pad(window_size)

        self.padded_image = padding(self.image)

        self.shape = self.image.shape

        self.length = self.image.flatten().shape

    def __getitem__(self, index):
        x_position = index // self.shape[2] + self.pad
        y_position = index % self.shape[2] + self.pad
        patch = self.padded_image[:, x_position - self.rad:x_position + self.rad + 1,
                y_position - self.rad:y_position + self.rad + 1]

        return patch

    def __len__(self):
        return self.length[0]


class BastetNet(nn.Module):
    """
    Patch-wise CNN network
    """
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(16 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes + 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def parse_args():
    """
    Parse command line arguments
    """
    argparser = ap.ArgumentParser(description="Image segmentation network")
    argparser.add_argument("-e", action="store",
                           dest="epochs", required=False, type=int,
                           help="Number of epochs to train", default=3)
    argparser.add_argument("-i", action="store", dest="input_file", type=str,
                           required=True,
                           help="Base image for segmentation")
    argparser.add_argument("-m", action="store", dest="mask", type=str,
                           required=True,
                           help="Mask images, encoded with indexed colors")
    argparser.add_argument("-s", dest="segment_files", action="store", type=str, nargs='+',
                           help="Images to segment using the trained network", required=False)
    argparser.add_argument("-sl", action="store", dest="save", type=str,
                           required=False, default="model.pt",
                           help="Path to store model")
    args = argparser.parse_args()

    return args


def main():
    """
    Train patch-wise cnn and segment images
    """
    args = parse_args()

    image = args.input_file
    mask = args.mask

    batch_size = 32

    dataset = ImageSet(image, mask, 28)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = BastetNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    num_epochs = args.epochs
    total_samples = len(dataset)
    n_iter = math.ceil(total_samples/batch_size)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for _, (input_batch, labels) in enumerate(dataloader):
            inputs = input_batch.to(device=device)
            labels = labels.to(device=device)

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'[{epoch + 1}] loss: {running_loss/n_iter:.3f}')

    torch.save(model.state_dict(), args.save)
    print(f"Saving trained model under: {args.save}\n")

    for file in args.segment_files:
        fullclassset = FullClassSet(file, 28)
        testloader = DataLoader(dataset=fullclassset, batch_size=batch_size, shuffle=False, num_workers=2)

        all_pixels = []

        with torch.no_grad():
            for data in testloader:
                # images, labels = data
                images = data.to(device=device)
                # calculate outputs by running images through the network
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                all_pixels.append(predicted)

        x = torch.tensor([[0, 0, 0], [255, 0, 0], [0, 0, 255]]).to(device)

        cat = torch.cat(all_pixels, dim=0)
        cat = F.embedding(cat, x).permute(1, 0)

        cat = cat.reshape((3, fullclassset.image.shape[1], fullclassset.image.shape[2])).type(torch.float)

        save_loc = f"{os.path.splitext(file)[0]}_segmented.tif"
        save_image(cat, save_loc)
        print(f"Segmented: {file} \n"
              f"Saving at: {save_loc}\n")

    return 0


if __name__ == '__main__':
    exitcode = main()
    sys.exit(exitcode)
