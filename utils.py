import transforms as T
import numpy as np
import torch

def get_transform(train, resolution, background_path):
    transforms = []

    # if square resolution, perform some aspect cropping
    # otherwise, resize to the resolution as specified

    if resolution[0] == resolution[1]:
        base_size = resolution[0] + 32 #520
        crop_size = resolution[0]      #480

        min_size = int((0.75 if train else 1.0) * base_size)
        max_size = int((1.25 if train else 1.0) * base_size)

        transforms.append(T.RandomResize(min_size, max_size))

        # during training mode, perform some data randomization
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
            transforms.append(T.RandomCrop(crop_size))
            transforms.append(T.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.4, probability = 0))
    else:



        #transforms.append(T.RandomResize2((resolution), 320))
        if train:
            transforms.append(T.BackgroundSubstitution(background_path=background_path))
            transforms.append(T.RandomHorizontalFlip(0.5))
            transforms.append(T.ColorJitter(brightness=0.2, contrast=0.3, saturation = 0.3, hue = 0.4, probability = 0.23))

    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

def collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets


def get_data_from_tensors(image, mask):
    image2 = image.permute(1,2,0).numpy()
    mask2 = mask.numpy()
    mask2 = mask2.astype(np.float32)

    return image2, mask2



if __name__ == '__main__':
    import torchvision

    model = torchvision.models.segmentation.__dict__['fcn_resnet50'](num_classes=2)
    params_to_optimize = [{'params': []}]

    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            params_to_optimize[0]['params'].append(param)



import torchvision.transforms as tr
def get_transform_train():
    transform = tr.Compose([
        tr.RandomHorizontalFlip(),
        tr.RandomRotation(30),
        tr.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        tr.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        tr.ToTensor(),
        tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform