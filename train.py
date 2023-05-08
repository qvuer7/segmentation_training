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
    'lr': [0.008],
    'momentum': [0.9],
    'weight_decay': [0.01],
    'L1_lambda': [0.001],
    'n_epochs': [50],
    'loss_weights': [[1.0,5.0], [1.0, 4.0]]
}
# min_image_sizes = [300]
# batch_sizes = [8]
# learning_rates = [0.005]
# momentums = [0.9]
# weight_decays = [0.01]
# L1_lambdas = [0.01]
# ns_epochs = [50]

n_classes = 2

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


# try:
#     os.mkdir(image_save_path)
#     os.mkdir(model_save_path)
#     os.mkdir(global_logdir_path)
#     print('dirs created')
# except Exception as e:
#     print(e)


def train_segmentor(params):
    min_image_size, batch_size, lr, momentum, weight_decay, L1_lambda, n_epochs, loss_weight = params
    job_path, checkpoints_path, images_path, log_path = create_training_job_folders(params, save_path = results_path)
    writer = SummaryWriter(log_dir=log_path)


    aspect_ratio = 640 / 480
    image_resolution = (min_image_size, int(min_image_size * aspect_ratio))

    train_transform = get_transform(train = True, resolution = image_resolution, background_path=background_path)
    test_transform  = get_transform(train = False, resolution = image_resolution, background_path=background_path)
    test_loader, train_loader = get_loaders(dataset_path=dataset_path, train_transform=train_transform, test_transform=test_transform,
                                            n_workers=n_workers, batch_size=batch_size)

    model, params_to_optimize = get_model(model_name = 'resnet50', n_classes=2)
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
                                     criterion=criterion, params=params_to_optimize, device = device)


        testCELoss, IOULoss, DiceLoss = evaluate(model=model, dataloader=test_loader, criterion=criterion,
                                            device=device, wghts=torch.tensor(loss_weight))

        writer.add_scalar(f"Loss/train", trainCELoss, epoch)
        writer.add_scalar(f"Loss/val", testCELoss, epoch)
        writer.add_scalar(f"Loss/IOU", IOULoss, epoch)
        writer.add_scalar(f"Loss/DICE", DiceLoss, epoch)

        print(f'TR : {trainCELoss}  |  VAL : {testCELoss}')
        print(f'D  : {DiceLoss}  |  IOU : {IOULoss}')

        IOULoss+=20
        if epoch == 1 or epoch % 15 == 0:
            vizualize(test_loader, model, epoch, save_path = images_path, device = device, every_n = 5)
            vizualize(train_loader, model, epoch, save_path=images_path, device = device, every_n = 200)
        if best_IOU < IOULoss:
            try:
                os.remove(os.path.join(checkpoints_path, f'model_{best_IOU}.pth'))
            except Exception as e:
                pass

            save_model(n_classes=n_classes, resolution = image_resolution, model = model,
                       IOULoss= IOULoss, checkpoint_paths= checkpoints_path )


            vizualize(test_loader, model, epoch, save_path=images_path, device=device, every_n=5)

            best_IOU = IOULoss





if __name__ == '__main__':
    param_combinations = list(itertools.product(*param_grid.values()))

    for params in param_combinations:
        train_segmentor(params = params)
    #     break
    # min_image_size, batch_size, lr, momentum, weight_decay, L1_lambda, n_epochs, loss_weight = params
    # job_path, checkpoints_path, images_path, log_path = create_training_job_folders(params, save_path=results_path)
    # writer = SummaryWriter(log_dir=log_path)
    #
    # aspect_ratio = 640 / 480
    # image_resolution = (min_image_size, int(min_image_size * aspect_ratio))
    # from utils import get_train_transform, get_test_transform
    #
    # train_transform = get_train_transform(resolution=image_resolution, background_path=background_path)
    # test_transform = get_test_transform(resolution=image_resolution)
    #
    # train_loader, test_loader = get_loaders(dataset_path=dataset_path, train_transform=train_transform,
    #                                         test_transform=test_transform,
    #                                         n_workers=n_workers, batch_size=batch_size)

    from PIL import Image
    from utils import get_image_label
    import matplotlib.pyplot as plt

    # image = Image.open(r'C:\Users\Andrii\PycharmProjects\segmentationTraining\segmentation_dataset_24_01\images\training\22_01_13_16_05_12.jpg')
    # mask  = Image.open(r'C:\Users\Andrii\PycharmProjects\segmentationTraining\segmentation_dataset_24_01\annotations\training\22_01_13_16_05_12.png')
    # i, m = train_transform(image, mask)
    # i, m = get_image_label(i, m)
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(i)
    # ax2.imshow(m)
    # plt.show()

    #
    # train_transform  =  [
    #     T.RandomAffine(),
    #     T.BackgroundSubstitution(background_path=background_path),
    #     T.Resize(image_resolution),
    #     T.RandomHorizontalFlip(0.5),
    #     T.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.4, probability=0.23),
    #     T.ToTensor(),
    #     T.Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225])
    #
    #
    # ]
    #
    # train_transform = T.Compose(train_transform)
    #
    # test_transforms = [
    #     T.Resize(image_resolution),
    #     T.ToTensor(),
    #     T.Normalize(mean=[0.485, 0.456, 0.406],
    #                 std=[0.229, 0.224, 0.225])
    # ]
    #
    # test_transforms = T.Compose(test_transforms)
    # print(f' test_transforms: {test_transforms}')
    #
    # test_loader, train_loader = get_loaders(dataset_path=dataset_path, train_transform=train_transform,
    #                                         test_transform=test_transform,
    #                                         n_workers=n_workers, batch_size=batch_size)
    # for idx, (images, labels) in enumerate(train_loader):
    #     break

    #
    #
    # #
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