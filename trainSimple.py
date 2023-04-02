import torchvision
import torch
from dataset import CustomSegmentation
from utils import get_transform, collate_fn, get_data_from_tensors
import cv2
import numpy as np
import torch.nn as nn
from tqdm import tqdm




device = torch.device('cpu')
n_classes = 2
n_epochs = 30
batch_size = 8
lr = 0.01
momentum = 0.9
weight_decay = 1e-4
n_workers = 0
image_save_path = r'C:\Users\Andrii\PycharmProjects\segmentationTraining\saved_images'
model_save_path = r'C:\Users\Andrii\PycharmProjects\segmentationTraining\pytorch-segmentation\model_1'
def train_one_epoch(model, dataloader, optimizer,criterion, device):
    model.train()
    tl = 0

    for c, (image, target) in enumerate(dataloader):


        image, target = image.to(device), target.to(device)
        out = model(image)
        loss = criterion(out, target)

        tl+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return tl/c

def evaluate(model, criterion, dataloader, device, epoch, save_path):
    model.eval()
    tl = 0
    saved = 0
    with torch.no_grad():
        for c, (image, target) in enumerate(dataloader):
            image, target = image.to(device), target.to(device)
            out = model(image)
            loss = criterion(out, target)
            tl+= loss.item()
            if (epoch % 3 == 0) and (saved == 0):
                image = image.squeeze()
                out = out.squeeze()
                i, m = get_data_from_tensors(image, out)
                cv2.imwrite(f'{save_path}\\{epoch}_mask.jpg',out)
                print('IMAGE SAVED')

    return tl/c



def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']



def main():
    model_names = sorted(name for name in torchvision.models.segmentation.__dict__
        if name.islower() and not name.startswith("__")
        and callable(torchvision.models.segmentation.__dict__[name]))

    weights = torchvision.models.ResNet50_Weights
    model = torchvision.models.segmentation.__dict__['fcn_resnet50'](num_classes = n_classes, weights_backbone = weights)

    params_to_optimize = [{'params': []}]

    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            params_to_optimize[0]['params'].append(param)




    train_transform = get_transform('train', resolution=(320,320))
    test_transform  = get_transform(False, resolution = (320,320))

    train_dataset = CustomSegmentation(root_dir = r'C:\Users\Andrii\PycharmProjects\segmentationTraining\segmentation_dataset_24_01'
                                       , image_set = 'train',
                                       transforms = train_transform)
    test_dataset  = CustomSegmentation(root_dir = r'C:\Users\Andrii\PycharmProjects\segmentationTraining\segmentation_dataset_24_01',
                                       image_set = 'val',
                                       transforms = test_transform)

    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = batch_size,num_workers = n_workers,
        collate_fn = collate_fn, drop_last = True, sampler = train_sampler)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size = 1,collate_fn = collate_fn,
        num_workers = n_workers, sampler = test_sampler)


    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=lr, momentum=momentum, weight_decay=weight_decay)

    model = model.to(device)


    for epoch in range(n_epochs):
        print(f'-'*10 + 'EPOCH: {i}')
        tr_loss = train_one_epoch(model = model, optimizer = optimizer,
                                  dataloader = train_loader, criterion=criterion,
                                  device = device)
        tl = evaluate(model = model, dataloader = test_loader, criterion=criterion,
                      device = device, epoch = epoch, save_path = image_save_path)

        print(f'TRAINING LOSS: {tr_loss}')
        print(f'TESTING  LOSS: {tl} ')
        if epoch % 10 == 0:
            torch.save({'model': model.state_dict(),
                        'num_classes': n_classes,
                        'resolution' : (320, 320),
                        'arch': 'fcn_resnet50'}, f'{model_save_path}/model_{epoch}.pth',
                       )

if __name__ == '__main__':
    main()








# for epoch in range(n_epochs):
#     l = train_one_epoch(model = model, optimizer = optimizer, criterion = criterion,
#                         dataloader = train_loader, device = device)
#     print(l)


# for idx, (image, mask) in enumerate(test_dataset):
#     break
#

# image, mask = get_data_from_tensors(image, mask)
# cv2.imshow('image', image)
# cv2.imshow('mask', mask)
# cv2.waitKey(0)


# torch.save({'model': model.state_dict(),
#             'num_classes': nclasses,
#             'resolution' : (320, 320),
#             'arch': 'fcn_resnet50'}, 'model_1/model_1.pth',
#            )
#

