import torchvision
import torch
from dataset import CustomSegmentation
from utils import get_transform, collate_fn, get_data_from_tensors
import cv2
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from resnet34.fcn_resnet import fcn_resnet34
from torch.utils.tensorboard import SummaryWriter
from utils import get_transform_train
import warnings
import os

warnings.filterwarnings("ignore")

device = torch.device('cuda')
learning_rates = [0.05, 0.08]
n_classes = 2
n_epochs = 40
batch_size = 6

momentum = 0.9
weight_decay = 0.01
L1_lambdas = [0.0001, 0.001]
n_workers = 2
image_save_path = '/content/images'
model_save_path = '/content/checkpoints/'
dataset_path    = '/content/drive/MyDrive/segmentation_dataset_24_01'

writer = SummaryWriter()

loss_type = 'BCE'

def train_one_epoch(model, dataloader, optimizer,criterion, device, params, L1_lambda):
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

        loss = criterion(out, target)
        loss += total_l1_loss
        loss.requires_grad_(True)
        tl+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return tl/c

def evaluate(model, criterion, dataloader, device, epoch, save_path):
    model.eval()
    print('EVALUATING')
    tl = 0
    saved = 0
    with torch.no_grad():
        for c, (image, target) in tqdm(enumerate(dataloader)):
            image, target = image.to(device), target.to(device)
            out = model(image)
            loss = criterion(out, target)
            tl+= loss.item()
            if (epoch % 10 == 0) and (saved == 0):
                image = image.squeeze().cpu()
                out = out['out'].squeeze().cpu()
                out = out.argmax(1)
                i, m = get_data_from_tensors(image, out)
                cv2.imwrite(f'{save_path}/{epoch}_image.jpg', i)
                cv2.imwrite(f'{save_path}/{epoch}_mask.jpg',m)
                print('IMAGE SAVED')
                saved = 1

    return tl/c



def CECriterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target)


    if len(losses) == 1:
        print('only one loss is used')
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

    if loss_type == 'BCE':
        criterion = BCECriterion
    else:
        criterion = CECriterion

    model_names = sorted(name for name in torchvision.models.segmentation.__dict__
        if name.islower() and not name.startswith("__")
        and callable(torchvision.models.segmentation.__dict__[name]))

    weights = torchvision.models.ResNet50_Weights
    model = torchvision.models.segmentation.__dict__['fcn_resnet50'](num_classes = n_classes, weights_backbone = weights)
    #model = fcn_resnet34(pretrained=False, progress=True, num_classes=2, aux_loss=False)
    params_to_optimize = [{'params': []}]

    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            params_to_optimize[0]['params'].append(param)





    train_transform = get_transform('train', resolution=(480,640))
    test_transform  = get_transform(False, resolution = (480,640))

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




    model = model.to(device)
    tr_loss = 0

    max_val_loss = torch.inf
    for L1_lambda in L1_lambdas:
        for lr in learning_rates:
            optimizer = torch.optim.SGD(
                params_to_optimize,
                lr=lr, momentum=momentum, weight_decay=weight_decay)
            print(f'L1: {L1_lambda} | LR: {lr} | best val: {max_val_loss}')
            print(f'training loss: {tr_loss}')
            for epoch in range(1,n_epochs+1):

                tr_loss = train_one_epoch(model = model, optimizer = optimizer,
                                          dataloader = train_loader, criterion=criterion,
                                          device = device, params = params_to_optimize, L1_lambda = L1_lambda)
                tl = evaluate(model = model, dataloader = test_loader, criterion=criterion,
                              device = device, epoch = epoch, save_path = image_save_path)

                writer.add_scalar(f"Loss/train_lr({lr})_l1({L1_lambda})", tr_loss, epoch)
                writer.add_scalar(f"Loss/val_lr({lr})_l1({L1_lambda})", tl, epoch)


                if  max_val_loss > tl:
                    try:
                        os.remove(model_best_path)
                    except Exception as e:
                        pass
                    max_val_loss = tl
                    torch.save({'model': model.state_dict(),
                                'num_classes': n_classes,
                                'resolution' : (320, 320),
                                'arch': 'fcn_resnet50'}, f'{model_save_path}/model_lr{str(lr)}_{str(epoch)}_l1{L1_lambda}.pth',
                               )
                    model_best_path = f'{model_save_path}/model_lr{str(lr)}_{str(epoch)}_l1{L1_lambda}.pth'

if __name__ == '__main__':
    main()
    # dataset_path = r'C:\Users\Andrii\PycharmProjects\segmentationTraining\segmentation_dataset_24_01'
    # train_transform = get_transform('train', resolution=(480,640))
    # test_transform  = get_transform(False, resolution = (480,640))
    #
    # train_dataset = CustomSegmentation(root_dir = dataset_path
    #                                    , image_set = 'train',
    #                                    transforms = train_transform)
    # train_sampler = torch.utils.data.RandomSampler(train_dataset)
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size = batch_size,num_workers = n_workers,
    #     collate_fn = collate_fn, drop_last = True, sampler = train_sampler)
    #
    # import matplotlib.pyplot as plt
    # for idx, (image_batch, label_batch) in enumerate(train_loader):
    #     break
    #
    #
    #
    #
    # weights = torchvision.models.ResNet50_Weights
    # model = torchvision.models.segmentation.__dict__['fcn_resnet50'](num_classes=n_classes, weights_backbone=weights)
    # with torch.no_grad():
    #     out = model(image_batch)
    #
    # CELoss = BCECriterion(out, label_batch)
    # print(CELoss.requires_grad)
    #

    #
    #
    #
    #
    #
    #
    #
    # c = 0
    # for image, label, los in zip(image_batch, out['out'], BCELoss):
    #     c+=1
    #     image = image.numpy()
    #     label =  label.argmax(dim=0)
    #     label = torch.sigmoid(label)
    #     loss = torch.sigmoid(los)
    #     loss = los.view(label.size()).numpy()
    #     image = image.transpose(1,2,0)
    #     min_value = image.min()
    #     max_value = image.max()
    #     new_min = 0
    #     new_max = 255
    #     image = (image - min_value) * (new_max / (max_value - min_value))
    #     image = image.astype(np.uint8)
    #     fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    #     ax1.imshow(image)
    #     ax2.imshow(label)
    #     ax3.imshow(loss)
    #     plt.savefig(save_figs_path + f'{c}.png')
    #
    #
    #




