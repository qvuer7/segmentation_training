import torch
from criterions import dice_coefficient, iou

def train_one_epoch(model, dataloader, optimizer,criterion, device, params, L1_lambda, wghts, n_classes):
    model.train()
    tl = 0
    for c, (image, target) in enumerate(dataloader):


        image, target = image.to(device), target.to(device)
        out = model(image)
        total_l1_loss = 0
        for v in params[0]['params']:
            total_l1_loss += torch.norm(v.data, 1)

        #total_l1_loss = total_l1_loss * L1_lambda

        loss = criterion(out, target, we = wghts, device = device, n_classes = n_classes)
        #loss += total_l1_loss

        loss += 1 - (iou(out['out'], target, n_classes)*0.5)
        loss += 1 - (dice_coefficient(out['out'], target, n_classes)*1)
        loss.requires_grad_(True)
        tl+=loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return tl/c



def evaluate(model, criterion, dataloader, device, wghts, n_classes):
    model.eval()
    print('EVALUATING')
    tl = 0
    io = 0
    dl = 0
    with torch.no_grad():
        for c, (image, target) in enumerate(dataloader):
            image, target = image.to(device), target.to(device)
            out = model(image)
            io += iou(out['out'], target, n_classes = n_classes)
            dl += dice_coefficient(out['out'], target, n_classes = n_classes)
            loss = criterion(out, target, we = wghts, device = device, n_classes = n_classes)
            tl+= loss.item()


    return tl/c, io/c, dl/c