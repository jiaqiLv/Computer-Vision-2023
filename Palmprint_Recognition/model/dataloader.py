from PIL import Image
import torchvision.transforms.functional as F
import scipy.misc
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10


class PalmDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = []
        self.targets = []

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.75, 1.25), ratio=(0.75, 1.25)),
            transforms.ColorJitter(0.25, 0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        self.transform_flip = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.75, 1.25), ratio=(0.75, 1.25)),
            transforms.ColorJitter(0.25, 0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip(1.0)
        ])

        file_path = './dataset/train.txt'

        with open(file_path) as f:
            lines = f.readlines()
            for line in lines:
                img, target = line.strip().split(' ')
                self.data.append(img)
                self.targets.append(int(target))

        self.size = len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if index < self.size:
            img, target = self.data[index], self.targets[index]
            img = self.transform(Image.open(img))
        else:
            img, target = self.data[index - self.size], self.targets[index - self.size] + 320
            img = self.transform_flip(Image.open(img))

        return img, target

    def __len__(self):
        return self.size * 2


class PalmTestSet(Dataset):
    def __init__(self, imgl, imgr):

        self.imgl_list = imgl
        self.imgr_list = imgr

        self.transform = transforms.Compose([
            transforms.Resize([224, 224], interpolation=F.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):

        imgl = self.transform(Image.open(self.imgl_list[index]))
        imgr = self.transform(Image.open(self.imgr_list[index]))

        return imgl, imgr

    def __len__(self):
        return len(self.imgl_list)


if __name__ == '__main__':
    pass
