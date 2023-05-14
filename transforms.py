import os
import torchvision
from torchvision import transforms as T
import torchvision.transforms.functional as F
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


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

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
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
        spear_coords = np.where(l == 1)



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


class BackgroundSubstitution():
    def __init__(self, background_path):
        self.rate = 1
        self.background_path = background_path
        self.background_photos = os.listdir(background_path)
    def __call__(self, image, target):
        p = torch.rand(1)

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



class RandomRotation(object):
    def __init__(self, min_angle = 30, max_angle = 75, rotation_probability = 0.35, expansion_probability = 0.99):
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.rotation_probability = rotation_probability
        self.expansion_probability = expansion_probability

    def __call__(self, image, target):
        if torch.rand(1) < self.rotation_probability:
            angle = int(torch.randint(self.min_angle, self.max_angle, (1,)))
            if torch.rand(1) < self.expansion_probability:
                image = image.rotate(angle, expand = False)
                target = target.rotate(angle, expand = False)
            else:
                image = image.rotate(angle, expand = False)
                target = target.rotate(angle, expand = False)
        return image, target







class RandomAffine(object):
    def __init__(self, degrees = 20.0, probability = 0.5, translate=None, scale=None, shear=None):
        self.probability = probability
        self.angle = degrees
        self.translation = [-3, 3]
        self.shear = [-1, 1]
        self.scale = [1, 1.5]
    def __call__(self, image, mask):
        if torch.rand(1) < self.probability:
            angle = float(torch.rand(1) * self.angle)
            translation = [int(torch.rand(1) * self.translation[0] + self.translation[1]),
                           int(torch.rand(1) * self.translation[0] + self.translation[1])]
            shear = [float(torch.rand(1) * self.shear[0] + self.shear[1]),
                     float(torch.rand(1) * self.shear[0] + self.shear[1])]
            scale = float(torch.rand(1)*0.5 +1)

            im = transforms.ToTensor()(image)
            ma = transforms.ToTensor()(mask)

            both_images = torch.cat((im, ma), 0)

            # Apply the transformations to both images simultaneously:

            transformed_images = F.affine(both_images, angle = angle, translate = translation, shear = shear, scale = scale)


            image = transformed_images[:3]

            mask = transformed_images[-1]
            mask = mask.unsqueeze(0)


            image = transforms.ToPILImage()(image)
            mask  = transforms.ToPILImage()(mask)




        return image, mask

if __name__ == '__main__':
    b = BackgroundSubstitution(background_path=r'C:\Users\Andrii\PycharmProjects\segmentationTraining\background\\')

    image = Image.open(r'C:\Users\Andrii\PycharmProjects\segmentationTraining\segmentation_dataset_24_01\images\training\22_01_13_16_04_22.jpg')
    mask  = Image.open(r'C:\Users\Andrii\PycharmProjects\segmentationTraining\segmentation_dataset_24_01\annotations\training\22_01_13_16_04_22.png')

    tran = []
    tran.append(BackgroundSubstitution(background_path=r'C:\Users\Andrii\PycharmProjects\segmentationTraining\background\\'))
    a  = Compose(tran)
    a(image, mask)

