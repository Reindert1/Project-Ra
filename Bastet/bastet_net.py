#!usr/bin/env python3

"""
"""

__author__ = "Skippybal"
__version__ = "0.1"

import math
import sys

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#from torchsummary import summary
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.set_num_threads(12)
# thread = torch.get_num_threads()
# print(thread)


class ImageSet(Dataset):

    def __init__(self, dataloc, labels, window_size):
        self.pad = window_size
        self.rad = window_size // 2
        transform = transforms.Grayscale()
        convert_tensor = transforms.ToTensor()
        self.image = convert_tensor(transform(Image.open(dataloc)))
        #self.image = convert_tensor(cv2.imread(dataloc, cv2.IMREAD_GRAYSCALE))

        # self.np_labels = cv2.imread(labels).flatten()
        # self.valid_pos = np.where(self.np_labels != 1)

        # self.palette =

        self.np_labels = np.asarray(Image.open(labels)).flatten()
        self.valid_pos = np.where(self.np_labels != 0)[0]

        #self.labels = torch.from_numpy(self.np_labels).type(torch.LongTensor)
        self.labels = torch.tensor(self.np_labels, dtype=torch.long)
        #self.labels = torch.nn.functional.one_hot(self.labels)

        #Image.open("data/indexed_new_mask.tif")

        #self.labels = torch.from_numpy(cv2.imread(labels)).flatten()

        padding = transforms.Pad(window_size)

        #self.shift = self.rad + window_size

        self.padded_image = padding(self.image)

        self.shape = self.image.shape

        self.length = self.image.flatten().shape


        # x = np.load("datasets/windows.npy")
        # y = np.load("datasets/labels.npy")
        # self.x = torch.from_numpy(x)
        # self.y = torch.squeeze(torch.from_numpy(y))
        # #print(self.y.shape)
        # self.n_samples = y.shape[0]


    def __getitem__(self, index):
        #print(index)
        #print(self.shape[2])
        #print(self.valid_pos)
        index = self.valid_pos[index]

        x_position = index // self.shape[2] + self.pad
        y_position = index % self.shape[2] + self.pad
        #y_position = (index - index // self.shape[2] * x_position) % self.shape[2] + self.pad
        # print(index // self.shape[2])
        # print((index - index // self.shape[2] * x_position))
        # print(index % self.shape[2])

        # x_position = 325 + self.pad
        # y_position = 174 + self.pad

        # patch = self.image[:, x_position-self.rad:x_position+self.rad+1, y_position-self.rad:y_position+self.rad+1]
        patch = self.padded_image[:, x_position - self.rad:x_position + self.rad + 1,
                y_position - self.rad:y_position + self.rad + 1]
        # patch = self.padded_image[:, x_position - self.shift:x_position + self.shift + 1,
        #         y_position - self.shift:y_position + self.shift + 1]
        # print(x_position-7, y_position-7)

        return patch, self.labels[index] #self.x[index], self.x[index]

    def __len__(self):
        return self.valid_pos.shape[0]
        #return self.length[0] #self.n_samples


class FullClassSet(Dataset):

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
        #return self.length[0] #self.n_samples


class BastetNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        #self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes + 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main(args):

    #dataset = ImageSet("data/larger_data.tif", "data/mask_larger_data_labels.tif", 25)
    #dataset = ImageSet("data/mask_larger_data_labels_redo.tif", "data/indexed_new_mask.png", 25)
    # dataset = ImageSet("data/train_small_r4_c7.tif", "data/train_small_r4_c7_indexed_labels.png", 28)
    # # print(dataset.image.unique())
    # # print(Image.open("data/mask_larger_data.tif").format )
    # # print(torch.from_numpy(cv2.imread("data/mask_larger_data.tif")).unique())
    # #print(dataset.image.shape)
    # #print(dataset.image.unique())
    # #print(dataset.valid_pos)
    # print(dataset.labels.unique())
    # #print(np.unique(np.asarray(Image.open("data/indexed_new_mask.tif"))))
    # #print(dataset.valid_pos)
    # print(np.where(dataset.labels == 0))
    #
    # save_image(dataset.image, 'BIN_ima3.png')
    # #save_image(dataset.image, 'redone_labels.png')
    # # print(dataset.labels.unique())
    #
    # #print(dataset.rad)
    #
    # #print(dataset[40000].shape)
    # # print(dataset[326].shape)
    # # save_image(dataset[326], 'BIN_ima.png')
    # # print(dataset.length)
    # # print(len(dataset))
    # #print(dataset.valid_pos.shape)
    #
    # patch, label = dataset[len(dataset)-1]
    # # print(label)
    # # print(patch)
    # save_image(patch, 'BIN_ima2.png')
    # save_image(dataset.image[:, 174, 325], 'BIN_ima2.png')
    #save_image(dataset[325 * 2 +1], 'BIN_ima2.png')


    # tensor = dataset[40000].cpu().numpy()
    # cv2.imwrite(tensor, "image.png")
    # first_data = dataset[0]
    # features, labels = first_data
    #print(features,labels)

    batch_size = 32

    dataset = ImageSet("data/train_small_r4_c7.tif", "data/train_small_r4_c7_indexed_mito.png", 28)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = BastetNet().to(device)
    # print(features.shape)
    # net(features.to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #print(model.parameters())
    #print(model)
    # summary(model, (6))
    criterion = nn.CrossEntropyLoss()

    num_epochs = 1
    total_samples = len(dataset)
    n_iter = math.ceil(total_samples/batch_size)
    print(dataset.valid_pos)
    print(n_iter)
    print(total_samples)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (input, labels) in enumerate(dataloader):
            #print(input.shape)
            #print(i)
            # print(input)
            # print(labels)
            inputs = input.to(device=device)
            labels = labels.to(device=device)
            #print(labels)

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            #print(outputs)
            #print(outputs, labels)
            loss = criterion(outputs, labels)
            #print(loss)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # if i % 2000 == 1999:  # print every 2000 mini-batches
            # if i % 2000 == 1999:
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1999:.3f}')
            #     running_loss = 0.0
                #print(torch.argmax(outputs, dim=1), labels)
        print(f'[{epoch + 1}] loss: {running_loss/n_iter:.3f}')
        #running_loss = 0.0

    fullclassset = FullClassSet("data/train_small_r4_c7.tif", 28)
    testloader = DataLoader(dataset=fullclassset, batch_size=batch_size, shuffle=False, num_workers=2)

    all = []

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            # images, labels = data
            images = data.to(device=device)
            #print(images.shape)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            all.append(predicted)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()

    x = torch.tensor([[0, 0, 0], [255, 0, 0], [0, 0, 255]]).to(device)

    # cat = torch.unsqueeze(torch.cat(all, dim=0), dim=1)
    cat = torch.cat(all, dim=0)
    print(cat.shape)
    cat = F.embedding(cat, x).permute(1, 0)
    print(cat.shape)
    print(cat)
    print(fullclassset.image.shape[1:])
    print(cat.reshape((3, fullclassset.image.shape[1], fullclassset.image.shape[2])).type(torch.float))
    cat = cat.reshape((3, fullclassset.image.shape[1], fullclassset.image.shape[2])).type(torch.float)

    # print(cat.shape)
    #cat = torch.unsqueeze(cat.reshape(fullclassset.image.shape[1:]).type(torch.float), 0)
    #cat = cat.reshape(fullclassset.image.shape).type(torch.float)


    # x = torch.FloatTensor([[0, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0]])
    # x = torch.tensor([[0, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0]])
    #cat = torch.embedding(cat, )
    # converted_tensor = torch.nn.functional.embedding(network_output, x).permute(0, 3, 1, 2)

    # print(cat.unique())
    print(cat.shape)
    # print(cat.reshape(fullclassset.image.shape).shape)

    # transform = T.ToPILImage()
    #
    # # convert the tensor to PIL image using above transform
    # img = transform(cat)
    #
    # # display the PIL image
    # #img.show()
    # #Image.save(img,)
    # img.save("org.png")


    save_image(cat, 'mitochondria.png')
    save_image(dataset.image, 'orginal.png')

    return 0


if __name__ == '__main__':
    exitcode = main(sys.argv)
    sys.exit(exitcode)