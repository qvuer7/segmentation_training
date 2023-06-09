from utils import get_model, get_transform, vizualize, save_model, create_training_job_folders
from dataset import get_loaders
from trainvalutils import train_one_epoch, evaluate
from criterions import *


from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import itertools
import torch
import warnings
import os


param_grid = {
    'min_image_size': [300],
    'batch_size': [6],
    'lr': [0.005, 0.01],
    'momentum': [0.9],
    'weight_decay': [0.0001, 0.001],
    'L1_lambda': [0.1],
    'n_epochs': [95],
    # 'loss_weights': [[1.0,2.5], [1.0, 3.5], [1.0, 5.0]],
    'loss_weights': [[1.0, 1.0]],
    'io_coffs': [2.0],
    'dice_coffs': [2.0]
}
m = 'resnet50'
# min_image_sizes = [300]
# batch_sizes = [8]
# learning_rates = [0.005]
# momentums = [0.9]
# weight_decays = [0.01]
# L1_lambdas = [0.01]
# ns_epochs = [50]

n_classes = 1

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
n_workers = 2 if torch.cuda.is_available() else 0




if torch.cuda.is_available():
    dataset_path = '/content/drive/MyDrive/segmentation_dataset_24_01'
    background_path = '/content/drive/MyDrive/background/'
    results_path   = '/content/result/'
else:
    dataset_path = r'C:\Users\Andrii\PycharmProjects\segmentationTraining\segmentation_dataset_24_01\\'
    background_path = r'C:\Users\Andrii\PycharmProjects\segmentationTraining\background\\'
    results_path = r'C:\Users\Andrii\PycharmProjects\segmentationTraining\results\\'

warnings.filterwarnings("ignore")



def train_segmentor(params):
    min_image_size, batch_size, lr, momentum, weight_decay, L1_lambda, n_epochs, loss_weight, io_coff, dice_coff = params
    job_path, checkpoints_path, images_path, log_path, train_check_images = create_training_job_folders(params, save_path = results_path)
    writer = SummaryWriter(log_dir=log_path)


    aspect_ratio = 640 / 480
    image_resolution = (min_image_size, int(min_image_size * aspect_ratio))

    train_transform = get_transform(train = True, resolution = image_resolution, background_path=background_path)
    test_transform  = get_transform(train = False, resolution = image_resolution, background_path=background_path)
    test_loader, train_loader = get_loaders(dataset_path=dataset_path, train_transform=train_transform, test_transform=test_transform,
                                            n_workers=n_workers, batch_size=batch_size)

    model, params_to_optimize = get_model(model_name = m, n_classes=n_classes)
    model = model.to(device)
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = CECriterion



    IOULoss = 0
    best_IOU = -torch.inf
    for epoch in tqdm(range(1,n_epochs + 1)):
        trainCELoss = train_one_epoch(model = model, dataloader=train_loader,L1_lambda=L1_lambda,
                                     wghts=torch.tensor(loss_weight), optimizer = optimizer,
                                     criterion=criterion, params=params_to_optimize, device = device,
                                      n_classes = n_classes)


        testCELoss, IOULoss, DiceLoss = evaluate(model=model, dataloader=test_loader, criterion=criterion,
                                            device=device, wghts=torch.tensor(loss_weight), n_classes=n_classes)

        writer.add_scalar(f"Loss/train", trainCELoss, epoch)
        writer.add_scalar(f"Loss/val", testCELoss, epoch)
        writer.add_scalar(f"Loss/IOU", IOULoss, epoch)
        writer.add_scalar(f"Loss/DICE", DiceLoss, epoch)

        print(f'TR : {trainCELoss}  |  VAL : {testCELoss}')
        print(f'D  : {DiceLoss}  |  IOU : {IOULoss}')


        if epoch == 1 or epoch % 15 == 0:
            vizualize(test_loader, model, epoch, save_path = images_path, device = device, every_n = 5, n_classes = n_classes)
            vizualize(train_loader, model, epoch, save_path=train_check_images, device = device, every_n = 25, n_classes = n_classes)
        if best_IOU < IOULoss:
            try:
                os.remove(os.path.join(checkpoints_path, f'model_{best_IOU}.pth'))
            except Exception as e:
                pass

            save_model(n_classes=n_classes, resolution = image_resolution, model = model,
                       IOULoss= IOULoss, checkpoint_paths= checkpoints_path, name = m )


            vizualize(test_loader, model, epoch, save_path=images_path, device=device, every_n=5, n_classes = n_classes)

            best_IOU = IOULoss





if __name__ == '__main__':
    param_combinations = list(itertools.product(*param_grid.values()))

    for params in param_combinations:

        train_segmentor(params = params)


    # model, parametrs = get_model(m, n_classes = n_classes)
    #
    # # save_model(n_classes = 2, model = model, name = m, resolution = (640, 480), IOULoss=12,
    # #            checkpoint_paths=r'C:\Users\Andrii\PycharmProjects\segmentationTraining\runs\\')
    # #     break
    #
    # min_image_size, batch_size, lr, momentum, weight_decay, L1_lambda, n_epochs, loss_weight, io_coff, dice_coff = params
    # job_path, checkpoints_path, images_path, log_path, _ = create_training_job_folders(params, save_path=results_path)
    # writer = SummaryWriter(log_dir=log_path)
    #
    # aspect_ratio = 640 / 480
    # image_resolution = (min_image_size, int(min_image_size * aspect_ratio))
    #
    # train_transform = get_transform(train=True, resolution=image_resolution, background_path=background_path)
    # test_transform = get_transform(train=False, resolution=image_resolution, background_path=background_path)
    # test_loader, train_loader = get_loaders(dataset_path=dataset_path, train_transform=train_transform,
    #                                         test_transform=test_transform,
    #                                         n_workers=n_workers, batch_size=batch_size)
    #
    # #
    # # vizualize(test_loader, model, 1,  r'C:\Users\Andrii\PycharmProjects\segmentationTraining\results\\', device, 5, 1)
    #
    # for idx, (images, labels) in enumerate(train_loader):
    #     break
    #
    # out = model(images)
    # print(out['out'].shape)
    # out = out['out']
    # import numpy as np
    # print(np.unique(out.detach().numpy()))
    # l = iou(out, labels, n_classes=n_classes)
    # print(print(l))
    # # mask = out[0].detach().squeeze()
    # # import matplotlib.pyplot as plt
    # # plt.imshow(mask)
    # # plt.show()
    #

    #
    # import matplotlib.pyplot as plt
    # from utils import get_image_label
    # import numpy as np
    # for i, m in zip(images, labels):
    #     i, m = get_image_label(i, m)
    #
    #     fig, (ax1, ax2) = plt.subplots(1, 2)
    #     ax1.imshow(i)
    #     ax2.imshow(m)
    #     plt.show()