import numpy as np
from PIL import Image
import random

import os
from torchvision import transforms as T
import torchvision.transforms.functional as F
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img
def pad_if_smaller2(img, sizex, sizey, fill =0):
    min_size = min(img.size)

    ow, oh = img.size
    padh = sizey - oh if oh < sizey else 0
    padw = sizex - ow if ow < sizex else 0
    img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.resize(image, self.size)
        target = F.resize(target, self.size, interpolation=Image.NEAREST)
        return image, target

class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=Image.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=0)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target

class RandomCrop2(object):
    def __init__(self, sizex, original_size):
        self.sizex = sizex
        self.original_size = original_size
    def __call__(self, image, target):
        aspect_ratio = self.original_size[0] / self.original_size[1]
        sizey = int(self.sizex / aspect_ratio)

        image = pad_if_smaller2(image, self.sizex, sizey)
        target = pad_if_smaller2(target, self.sizex,sizey, fill =  0)

        crop_params = T.RandomCrop.get_params(image, (self.sizex, sizey))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target

class RandomResize2(object):
    def __init__(self, original_size, min_edge_size):
        self.original_size = original_size
        self.min_edge_size = min_edge_size

    def __call__(self, image, label):
        aspect_ratio = self.original_size[0]/self.original_size[1]


        max_target_size = int(self.min_edge_size/aspect_ratio)
        image = F.resize(image, (self.min_edge_size, max_target_size))
        target = F.resize(label, (self.min_edge_size, max_target_size), interpolation=Image.NEAREST)

        return image, target
class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.asarray(target), dtype=torch.int64)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target




class ColorJitter:
    def __init__(self, brightness, contrast, saturation, hue, probability):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.probability = probability
    def __call__(self, image, label):
        if torch.rand(1) <  self.probability:

            if self.brightness > 0:
                image = transforms.functional.adjust_brightness(image, 1.0 + random.uniform(-self.brightness, self.brightness))

            if self.contrast > 0:
                image = transforms.functional.adjust_contrast(image, 1.0 + random.uniform(-self.contrast, self.contrast))

            if self.saturation > 0:
                image = transforms.functional.adjust_saturation(image, 1.0 + random.uniform(-self.saturation, self.saturation))

            if self.hue > 0:
                image = self.apply_hue(image,label,  self.hue)

        return image, label

    @staticmethod
    def apply_hue(image,label,  hue_factor):
        if torch.rand(1) < 0.5:
            hue_factor = -hue_factor


        image = image.convert("HSV")
        h, s, v = image.split()
        l = np.asarray(label)
        spear_coords = np.where(l == 255)



        np_h = np.array(h)
        ori = np_h.copy()
        np_h = np.mod(np_h.astype(np.int16) + np.int16(hue_factor * 255), 256).astype(np.uint8)
        if np.any(spear_coords[0]):
            spear_coords_x, spear_coords_y = spear_coords[0], spear_coords[1]
            np_h[spear_coords_x, spear_coords_y] = ori[spear_coords_x, spear_coords_y]
        h = Image.fromarray(np_h, "L")

        image = Image.merge("HSV", (h, s, v))
        image = image.convert("RGB")


        return image


class BackgroundSubstitution(object):
    def __init__(self, background_path ):
        self.rate = 0.9
        self.background_path = background_path
        self.background_photos = os.listdir(background_path)
    def __call__(self, image, target):
        p = torch.randn(1)
        if p < self.rate:
            v = np.random.randint(1, len(self.background_photos) - 2)
            self.background = Image.open(self.background_path + self.background_photos[v])

            self.background = np.array(self.background.resize(image.size))
            t = np.asarray(target)
            b = np.where(t == 1)
            x, y = b[0], b[1]
            i = np.asarray(image)
            self.background[x, y] = i[x,y]

            self.background = Image.fromarray(self.background)
            return self.background, target
        else:
            return image, target


if __name__ == '__main__':
    b = BackgroundSubstitution(background_path=r'C:\Users\Andrii\PycharmProjects\segmentationTraining\background\\')
    image = cv2.imread(r'C:\Users\Andrii\PycharmProjects\segmentationTraining\segmentation_dataset_24_01\images\training\22_01_13_16_04_22.jpg')
    mask = cv2.imread(r'C:\Users\Andrii\PycharmProjects\segmentationTraining\segmentation_dataset_24_01\annotations\training\22_01_13_16_04_22.png')

    i, m = b(image, mask)
    import matplotlib.pyplot as plt

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    ax1.imshow(image)
    ax2.imshow(mask)
    ax3.imshow(i)
    ax4.imshow(m)
    plt.show()





