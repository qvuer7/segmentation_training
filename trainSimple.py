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

warnings.filterwarnings("ignore")

device = torch.device('cuda')
learning_rates = [0.0001, 0.001, 0.01]
n_classes = 2
n_epochs = 31
batch_size = 8
lr = 0.01
momentum = 0.9
weight_decay = 0.01
L1_lambda = 0.0001
n_workers = 2
image_save_path = '/content/images'
model_save_path = '/content/checkpoints/'
dataset_path    = '/content/drive/MyDrive/segmentation_dataset_24_01'
writer = SummaryWriter()
def train_one_epoch(model, dataloader, optimizer,criterion, device, params):
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
        tl+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return tl/c

def evaluate(model, criterion, dataloader, device, epoch, save_path):
    model.eval()
    print('EVALUATING')
    tl = 0
    saved = 4
    with torch.no_grad():
        for c, (image, target) in tqdm(enumerate(dataloader)):
            image, target = image.to(device), target.to(device)
            out = model(image)
            loss = criterion(out, target)
            tl+= loss.item()
            if (epoch % 3 == 0) and (saved == 0):
                image = image.squeeze().cpu()
                out = out['out'].squeeze().cpu()
                out = out.argmax(1)
                i, m = get_data_from_tensors(image, out)
                cv2.imwrite(f'{save_path}/{epoch}_image.jpg', i)
                cv2.imwrite(f'{save_path}/{epoch}_mask.jpg',m)
                print('IMAGE SAVED')
                saved = 1

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
    #model = fcn_resnet34(pretrained=False, progress=True, num_classes=2, aux_loss=False)
    params_to_optimize = [{'params': []}]

    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            params_to_optimize[0]['params'].append(param)




    #train_transform = get_transform('train', resolution=(320,320))
    train_transform = get_transform_train()
    test_transform  = get_transform(False, resolution = (320,320))

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

    max_val_loss = torch.inf

    for lr in learning_rates:
        optimizer = torch.optim.SGD(
            params_to_optimize,
            lr=lr, momentum=momentum, weight_decay=weight_decay)
        print(F'STARTING TRAINING WITH {lr}')
        for epoch in range(1,n_epochs+1):
            print(f'-'*10 + f'EPOCH: {epoch}')
            tr_loss = train_one_epoch(model = model, optimizer = optimizer,
                                      dataloader = train_loader, criterion=criterion,
                                      device = device, params = params_to_optimize)
            tl = evaluate(model = model, dataloader = test_loader, criterion=criterion,
                          device = device, epoch = epoch, save_path = image_save_path)
            writer.add_scalar(f"Loss/train_{lr}", tr_loss, epoch)
            writer.add_scalar(f"Loss/val_{lr}", tl, epoch)

            print(f'TRAINING LOSS: {tr_loss}')
            print(f'TESTING  LOSS: {tl} ')
            if  max_val_loss > tl:
                max_val_loss = tl
                torch.save({'model': model.state_dict(),
                            'num_classes': n_classes,
                            'resolution' : (320, 320),
                            'arch': 'fcn_resnet50'}, f'{model_save_path}/model_{str(lr)}.pth',
                           )

if __name__ == '__main__':
    main()






