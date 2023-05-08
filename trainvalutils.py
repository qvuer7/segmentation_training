import torch
from criterions import dice_coefficient, iou

def train_one_epoch(model, dataloader, optimizer,criterion, device, params, L1_lambda, wghts, io_cof, dice_cof):
    model.train()
    tl = 0
    for c, (image, target) in enumerate(dataloader):


        image, target = image.to(device), target.to(device)
        out = model(image)
        total_l1_loss = 0
        for v in params[0]['params']:
            total_l1_loss += torch.norm(v.data, 1)

        total_l1_loss = total_l1_loss * L1_lambda

        loss = criterion(out, target, we = wghts, device = device)
        loss += total_l1_loss
        loss += (iou(out['out'], target)*io_cof)
        loss += (dice_coefficient(out['out'], target)*dice_cof)
        loss.requires_grad_(True)
        tl+=loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return tl/c



def evaluate(model, criterion, dataloader, device, wghts):
    model.eval()
    print('EVALUATING')
    tl = 0
    io = 0
    dl = 0
    with torch.no_grad():
        for c, (image, target) in enumerate(dataloader):
            image, target = image.to(device), target.to(device)
            out = model(image)
            io += iou(out['out'], target)
            dl += dice_coefficient(out['out'], target)
            loss = criterion(out, target, we = wghts, device = device)
            tl+= loss.item()


    return tl/c, io/c, dl/c