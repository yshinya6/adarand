import math
import os

import torchvision.datasets
import torchvision.transforms as T

default_mean = [0.5, 0.5, 0.5]
default_std = [0.5, 0.5, 0.5]


def cls_test_transforms(size, mean, std):
    upper_size = int(math.pow(2, math.ceil(math.log2(size))))
    return T.Compose([
        T.Resize(upper_size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])


def cls_train_transforms(size, mean, std):
    return T.Compose([
        T.RandomResizedCrop(size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])


class GenericDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, size=224, test=False):
        train = not test
        mean, std = (default_mean, default_std)
        if train:
            split_data_dir = os.path.join(root, "train")
            transform = cls_train_transforms(size=size, mean=mean, std=std)
        else:
            split_data_dir = os.path.join(root, "test")
            transform = cls_test_transforms(size=size, mean=mean, std=std)
        super(GenericDataset, self).__init__(root=split_data_dir, transform=transform)


class StanfordCars(GenericDataset):
    def __init__(self, root="/dataset/StanfordCars", size=224, test=False):
        super(StanfordCars, self).__init__(root, size, test)


class Birds(GenericDataset):
    def __init__(self, root="/dataset/CUB-200-2011", size=224, test=False):
        super(Birds, self).__init__(root, size, test)


class DTD(GenericDataset):
    def __init__(self, root="/dataset/DTD", size=224, test=False):
        super(DTD, self).__init__(root, size, test)


class Aircraft(GenericDataset):
    def __init__(self, root="/dataset/FGVC-Aircraft", size=224, test=False):
        super(Aircraft, self).__init__(root, size, test)


class Flower(GenericDataset):
    def __init__(self, root="/dataset/OxfordFlower102", size=224, test=False):
        super(Flower, self).__init__(root, size, test)


class Pets(GenericDataset):
    def __init__(self, root="/dataset/OxfordPets", size=224, test=False):
        super(Pets, self).__init__(root, size, test)
