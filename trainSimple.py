import matplotlib.pyplot as plt
import torchvision
import torch
from dataset import CustomSegmentation
from utils import get_transform, collate_fn, get_data_from_tensors, get_image
import cv2
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from resnet34.fcn_resnet import fcn_resnet34
from torch.utils.tensorboard import SummaryWriter
from utils import get_transform_train
import warnings
import os
import torch.nn.functional as F

warnings.filterwarnings("ignore")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
learning_rates = [0.005]
n_classes = 2
n_epochs = 35
batch_size = 6

momentum = 0.9
weight_decay = 0.01
L1_lambdas = [0.01]
n_workers = 2
image_save_path = '/content/images'
model_save_path = '/content/checkpoints/'
dataset_path    = '/content/drive/MyDrive/segmentation_dataset_24_01'
background_path = '/content/drive/MyDrive/background/'
# model_save_path = r'C:\Users\Andrii\PycharmProjects\segmentationTraining\\'
# dataset_path = r'C:\Users\Andrii\PycharmProjects\segmentationTraining\segmentation_dataset_24_01\\'
# image_save_path = r'C:\Users\Andrii\PycharmProjects\segmentationTraining\temp_images'
# background_path = r'C:\Users\Andrii\PycharmProjects\segmentationTraining\background\\'
writer = SummaryWriter()
w = torch.tensor([0.5, 4.0])
ws = [[1.0, 3.0] , [1.0,4.0], [1.0,5.0]]
loss_type = 'CE'


def vizualize(dataloader, model,  epoch, save_path):
    for c, (image, target) in tqdm(enumerate(dataloader)):
        if c%6 == 0:
            image, target = image.to(device), target.to(device)
            with torch.no_grad():
                out = model(image)
            image = image.squeeze().cpu()

            out = out['out'].squeeze().cpu()
            i = image[0].permute(1, 2, 0).numpy()
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
            plt.savefig(f'{save_path}/{epoch}_{c}.jpg')
            print('IMAGE SAVED')

def train_one_epoch(model, dataloader, optimizer,criterion, device, params, L1_lambda, wghts):
    model.train()
    tl = 0
    print('TRAINING')
    for c, (image, target) in tqdm(enumerate(dataloader)):


        image, target = image.to(device), target.to(device)
        out = model(image)
        total_l1_loss = 0
        for v in params[0]['params']:
            total_l1_loss += torch.norm(v.data, 1)

        total_l1_loss = total_l1_loss * L1_lambda

        loss = criterion(out, target, we = wghts)
        loss += total_l1_loss
        loss.requires_grad_(True)
        tl+=loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return tl/c



def dice_coefficient(prediction, target):
    smooth = 1e-5  # Small constant to avoid division by zero
    threshold = 0.5  # Threshold for converting network output to binary mask
    with torch.no_grad():
        predicted_class = torch.argmax(prediction, dim=1)  # Get the predicted class index
        prediction_binary = (predicted_class == 1).float()
        intersection = torch.sum(prediction_binary * target)
        union = torch.sum(prediction_binary) + torch.sum(target)
        dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()

def iou(prediction, target):
    smooth = 1e-5  # Small constant to avoid division by zero
    threshold = 0.5  # Threshold for converting network output to binary mask
    with torch.no_grad():
        predicted_class = torch.argmax(prediction, dim=1)  # Get the predicted class index
        prediction_binary = (predicted_class == 1).float()
        intersection = torch.sum(prediction_binary * target)
        union = torch.sum(prediction_binary) + torch.sum(target) - intersection
        iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def evaluate(model, criterion, dataloader, device, wghts):
    model.eval()
    print('EVALUATING')
    tl = 0
    io = 0
    dl = 0
    with torch.no_grad():
        for c, (image, target) in tqdm(enumerate(dataloader)):
            image, target = image.to(device), target.to(device)
            out = model(image)
            io += iou(out['out'], target)
            dl += dice_coefficient(out['out'], target)
            loss = criterion(out, target, we = wghts)
            tl+= loss.item()


    return tl/c, io/c, dl/c



def CECriterion(inputs, target, we):
    losses = {}

    weight = we.to(device)
    criterion = nn.CrossEntropyLoss(weight = weight)
    for name, x in inputs.items():
        # losses[name] = nn.functional.cross_entropy(x, target)
        losses[name] = criterion(x, target)


    if len(losses) == 1:

        return losses['out']

    return losses['out'] + 0.5 * losses['aux']

def BCECriterion(out, targets):
    predictions = out['out']

    batch_size, num_classes, height, width = predictions.size()

    # Apply argmax to get the segmentation mask
    segmentation_mask = predictions.argmax(dim=1)  # Shape: [batch_size, height, width]

    # Flatten the segmentation mask, output, and target tensors
    flattened_segmentation_mask = segmentation_mask.view(batch_size, -1)
    flattened_segmentation_mask = torch.sigmoid(flattened_segmentation_mask)
    flattened_target = targets.view(batch_size, -1).float()

    criterion = nn.BCELoss(reduction='mean')

    loss = criterion(flattened_segmentation_mask, flattened_target)

    # Optionally compute the average loss across batches and samples
    average_loss = loss.mean()


    return average_loss




def main():
    try:
        os.mkdir(image_save_path)
        os.mkdir(model_save_path)
        print('dirs created')
    except Exception as e:
        print(e)
    # if loss_type == 'BCE':
    #     criterion = BCECriterion
    # else:
    #     criterion = CECriterion
    criterion = CECriterion
    model_names = sorted(name for name in torchvision.models.segmentation.__dict__
        if name.islower() and not name.startswith("__")
        and callable(torchvision.models.segmentation.__dict__[name]))

    # weights = torchvision.models.ResNet50_Weights
    # model = torchvision.models.segmentation.__dict__['fcn_resnet50'](num_classes = n_classes, weights_backbone = weights)
    # #model = fcn_resnet34(pretrained=False, progress=True, num_classes=2, aux_loss=False)
    # params_to_optimize = [{'params': []}]
    #
    # for name, param in model.named_parameters():
    #     if 'backbone' in name:
    #         param.requires_grad = False
    #     else:
    #         param.requires_grad = True
    #         params_to_optimize[0]['params'].append(param)





    train_transform = get_transform('train', resolution=(480,640), background_path=background_path)
    test_transform  = get_transform(False, resolution = (480,640), background_path=background_path)

    train_dataset = CustomSegmentation(root_dir = dataset_path
                                       , image_set = 'train',
                                       transforms = train_transform)
    test_dataset  = CustomSegmentation(root_dir = dataset_path,
                                       image_set = 'val',
                                       transforms = test_transform)

    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = batch_size,num_workers = n_workers,
        collate_fn = collate_fn, drop_last = True, sampler = train_sampler)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size = batch_size,collate_fn = collate_fn,
        num_workers = n_workers, sampler = test_sampler, drop_last = True)




    # model = model.to(device)
    tr_loss = 0

    max_val_loss = -torch.inf
    for L1_lambda in L1_lambdas:
        for lr in learning_rates:
            for w in ws:

                weights = torchvision.models.ResNet50_Weights
                model = torchvision.models.segmentation.__dict__['fcn_resnet50'](num_classes=n_classes,weights_backbone=weights)
                # model = fcn_resnet34(pretrained=False, progress=True, num_classes=2, aux_loss=False)
                params_to_optimize = [{'params': []}]

                for name, param in model.named_parameters():
                    if 'backbone' in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                        params_to_optimize[0]['params'].append(param)
                model = model.to(device)
                optimizer = torch.optim.SGD(
                    params_to_optimize,
                    lr=lr, momentum=momentum, weight_decay=weight_decay)
                print(f'L1: {L1_lambda} | LR: {lr} | best val: {max_val_loss}')
                print(f'w : {w}')

                for epoch in range(1,n_epochs+1):

                    tr_loss = train_one_epoch(model = model, optimizer = optimizer,
                                              dataloader = train_loader, criterion=criterion,
                                              device = device, params = params_to_optimize, L1_lambda = L1_lambda, wghts=torch.tensor(w))
                    tl, io, dl = evaluate(model = model, dataloader = test_loader, criterion=criterion,
                                  device = device, wghts=torch.tensor(w) )

                    writer.add_scalar(f"Loss/train_B({w[0]})_A({w[1]})", tr_loss, epoch)
                    writer.add_scalar(f"Loss/val_B({w[0]})_A({w[1]})", tl, epoch)
                    writer.add_scalar(f"Loss/IOU({w[0]})_A({w[1]})", io, epoch)
                    writer.add_scalar(f"Loss/DICE({w[0]})_A({w[1]})", dl, epoch)


                    print(f'TR : {tr_loss}  |  VAL : {tl}')
                    print(f'D  : {dl}  |  IOU : {io}')

                    if  max_val_loss < io:
                        try:
                            os.remove(model_best_path)
                        except Exception as e:
                            pass
                        try:
                            os.mkdir(image_save_path + f'/model_B({w[0]})_A({w[1]})_l{round(io,4)}')
                        except Exception as e:
                            print(e)

                        vizualize(dataloader = test_loader, model = model, epoch = epoch, save_path=image_save_path + f'/model_B({w[0]})_A({w[1]})_l{round(io,4)}')
                        max_val_loss = io
                        torch.save({'model': model.state_dict(),
                                    'num_classes': n_classes,
                                    'resolution' : (480, 640),
                                    'arch': 'fcn_resnet50'}, f'{model_save_path}/model_B({w[0]})_A({w[1]})_loss{round(io,4)}.pth',
                                   )
                        model_best_path = f'{model_save_path}/model_B({w[0]})_A({w[1]})_loss{round(io,4)}.pth'

if __name__ == '__main__':
    main()
    # dataset_path = r'C:\Users\Andrii\PycharmProjects\segmentationTraining\segmentation_dataset_24_01'
    # background_path = r'C:\Users\Andrii\PycharmProjects\segmentationTraining\background\\'
    # train_transform = get_transform('train', resolution=(480,640), background_path=background_path)
    # test_transform  = get_transform(False, resolution = (480,640), background_path=background_path)
    #
    # train_dataset = CustomSegmentation(root_dir = dataset_path
    #                                    , image_set = 'train',
    #                                    transforms = train_transform)
    # test_dataset  = CustomSegmentation(root_dir = dataset_path,
    #                                    image_set = 'val',
    #                                    transforms = test_transform)
    #
    # train_sampler = torch.utils.data.RandomSampler(train_dataset)
    # test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    #
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size = 2,num_workers = n_workers,
    #     collate_fn = collate_fn, drop_last = True, sampler = train_sampler)
    #
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size = 6,collate_fn = collate_fn,
    #     num_workers = n_workers, sampler = test_sampler, drop_last = True)
    # for c, (images, labels) in enumerate(test_loader):
    #     break
    #
    #
    #
    # weights = torchvision.models.ResNet50_Weights
    # model = torchvision.models.segmentation.__dict__['fcn_resnet50'](num_classes=n_classes, weights_backbone=weights)
    #
    #
    # checkpoint = torch.load(r'C:\Users\Andrii\PycharmProjects\segmentationTraining\models\resnet50_weights\model_lr0.005_24_l10.2_loss0.0248.pth', map_location = device)
    #
    # model.load_state_dict(checkpoint['model'])
    # model.eval()
    # with torch.no_grad():
    #     out = model(images)
    #
    # dc = dice_coefficient(out['out'], labels)
    # i  = iou(out['out'], labels)
    # print(dc)
    # print(i)
    # # n = 1
    # # out = out['out'][n]
    # # label = labels[n].detach().numpy()
    # #
    # # out_mask = out.argmax(dim = 0).numpy()
    # # print(np.unique(out_mask))
    # # ori = get_image(images[n])
    # # fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    # # ax1.imshow(ori)
    # # ax2.imshow(label)
    # # ax3.imshow(out_mask)
    # # plt.show()
    # # for image, label in zip(images, labels):
    # #     image = image.permute(1,2,0).numpy()
    # #     label = label.numpy()
    # #     min_value = image.min()
    # #     max_value = image.max()
    # #     new_min = 0
    # #     new_max = 255
    # #     image = (image - min_value) * (new_max / (max_value - min_value))
    # #     image = image.astype(np.uint8)
    # #     fig, (ax1, ax2) = plt.subplots(1,2)
    # #     ax1.imshow(image)
    # #     ax2.imshow(label)
    # #     plt.show()
    #
