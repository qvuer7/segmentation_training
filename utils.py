import transforms as T
import numpy as np
import torch
from resnet34.fcn_resnet import fcn_resnet34
import torchvision
import matplotlib.pyplot as plt
import os
import torchvision.transforms as TF

def get_transform(train, resolution, background_path):
    trans = []


    if resolution[0] == resolution[1]:
        base_size = resolution[0] + 32 #520
        crop_size = resolution[0]      #480

        min_size = int((0.75 if train else 1.0) * base_size)
        max_size = int((1.25 if train else 1.0) * base_size)



        if train:

            trans.append(T.BackgroundSubstitution(background_path=background_path))
            trans.append(T.RandomResize(min_size, max_size))
            trans.append(T.RandomHorizontalFlip(0.5))
            trans.append(T.RandomCrop(crop_size))
            trans.append(T.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.4, probability = 0.23))
        else:
            trans.append(T.RandomResize(min_size, max_size))
    else:



        #transforms.append(T.RandomResize2((resolution), 320))
        if train:
            # transforms.append(T.RandomRotation())
            trans.append(T.RandomAffine())
            trans.append(T.BackgroundSubstitution(background_path=background_path))
            trans.append(T.Resize(resolution))
            trans.append(T.RandomHorizontalFlip(0.5))
            trans.append(T.ColorJitter(brightness=0.2, contrast=0.3, saturation = 0.3, hue = 0.4, probability = 0.23))
        else:
            trans.append(T.Resize(resolution))
    trans.append(T.ToTensor())
    trans.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))

    return T.Compose(trans)


def get_train_transform(resolution, background_path):
    # train_trans = []
    #
    # train_trans.append(T.RandomAffine())
    # train_trans.append(T.BackgroundSubstitution(background_path=background_path))
    # train_trans.append(T.Resize(resolution))
    # train_trans.append(T.RandomHorizontalFlip(0.5))
    # train_trans.append(T.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.4, probability=0.23))
    #
    # train_trans.append(T.ToTensor())
    # train_trans.append(T.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225]))

    train_trans = [
        T.RandomHorizontalFlip(0.5),
        T.RandomAffine(),
        T.BackgroundSubstitution(background_path=background_path),
        T.Resize(resolution),
        T.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.4, probability=0.23),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


    ]

    return T.Compose(train_trans)

def get_test_transform(resolution):

    test_trans = [
        T.Resize(resolution),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ]


    return T.Compose(test_trans)

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

def get_image_label(image_tensor, label):
    predicted_mask = image_tensor.detach().permute(1, 2, 0).numpy()
    min_value = predicted_mask.min()
    max_value = predicted_mask.max()
    new_min = 0
    new_max = 255
    predicted_mask = (predicted_mask - min_value) * (new_max / (max_value - min_value))
    predicted_mask = predicted_mask.astype(np.uint8)
    mask = label.numpy()
    return predicted_mask, mask



def get_model(model_name = 'resnet50', n_classes = 1):
    if model_name == 'resnet50':
        weights = torchvision.models.ResNet50_Weights
        model = torchvision.models.segmentation.__dict__['fcn_resnet50'](num_classes=n_classes,
                                                                         weights_backbone=weights)
    elif model_name == 'resnet34':
        model = fcn_resnet34(pretrained=False, progress=True, num_classes=2, aux_loss=False)

    else :
        weights =  torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
        model = torchvision.models.segmentation.__dict__['deeplabv3_resnet50'](num_classes = n_classes,
                                                                    weights = weights)

    params_to_optimize = [{'params': []}]
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            params_to_optimize[0]['params'].append(param)

    return model, params_to_optimize

def save_model(n_classes, checkpoint_paths, model, IOULoss, resolution, name):
    if name == 'resnet50':
        arch = 'fcn_resnet50'
    elif name == 'resnet34':
        arch = 'fcn_resnet34'
    else:
        arch = 'deeplabv3_resnet50'
    path = os.path.join(checkpoint_paths, f'model_{IOULoss}.pth')
    torch.save({'model': model.state_dict(),
                'num_classes': n_classes,
                'resolution': resolution,
                'arch': arch}, path,
               )
    print(f'model saved: {path}')

def vizualize(dataloader, model,  epoch, save_path, device, every_n, n_classes):
    for c, (image, target) in enumerate(dataloader):
        if c%every_n == 0:
            image, target = image.to(device), target.to(device)
            with torch.no_grad():
                out = model(image)
            image = image.squeeze().cpu()

            out = out['out'].squeeze().cpu()
            i = image[0].permute(1, 2, 0).numpy()
            if n_classes == 1:
                m = out[0].numpy().squeeze()
            else:
                m = out[0].argmax(dim=0).numpy()

            min_value = i.min()
            max_value = i.max()
            new_min = 0
            new_max = 255
            i = (i - min_value) * (new_max / (max_value - min_value))
            i = i.astype(np.uint8)

            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(i)
            ax2.imshow(m)
            path = os.path.join(save_path, f'{epoch}_{c}.jpg')
            plt.savefig(path)
            print('IMAGE SAVED')


def create_training_job_folders(params, save_path):
    min_image_size, batch_size, lr, momentum, weight_decay, L1_lambda, n_epochs, loss_weight, io_cof, dice_cof = params

    model_name = f"model_size{min_image_size}_batch{batch_size}_lr{lr}_momentum{momentum}_wd{weight_decay}_L1{L1_lambda}_epochs{n_epochs}_weights{loss_weight}_iou_{io_cof}_dce_{dice_cof}"
    job_path = os.path.join(save_path, model_name)
    images_path = os.path.join(job_path, 'images')
    checkpoints_path = os.path.join(job_path, 'checkpoints')
    log_path = os.path.join(job_path, 'logs')
    images_train_path = os.path.join(images_path, 'train')
    try:
        os.mkdir(save_path)
    except Exception as e:
        print(e)

    try:
        os.mkdir(job_path)
    except Exception as e:
        print(e)

    try:
        os.mkdir(images_path)
    except Exception as e:
        print(e)

    try:
        os.mkdir(checkpoints_path)
    except Exception as e:
        print(e)

    try:
        os.mkdir(log_path)
    except Exception as e:
        print(e)

    try:
        os.mkdir(images_train_path)
    except Exception as e:
        print(e)

    return job_path, checkpoints_path, images_path, log_path, images_train_path



if __name__ == '__main__':
    # Load the pretrained model
    weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    pretrained = torchvision.models.segmentation.__dict__['deeplabv3_resnet50'](
                                                                           weights=weights)

    target_model = torchvision.models.segmentation.__dict__['deeplabv3_resnet50'](num_classes = 2)

    target_model_dict = target_model.state_dict()
    target_model_dict2 = {k: v for k, v in pretrained.state_dict().items() if k in target_model_dict}

    target_model_dict.update(target_model_dict2)
    target_model.load_state_dict(target_model_dict)
    # target_model.backbone.load_state_dict(pretrained.backbone.state_dict())
    # target_model.classifier.load_state_dict(pretrained.classifier.state_dict())





