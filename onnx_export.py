#
# converts a saved PyTorch model to ONNX format
#
import os
import argparse

import torch
import torchvision.models as models
from resnet34.fcn_resnet import fcn_resnet34
def main():
    checkpoint_path = r'C:\Users\Andrii\Downloads\lr0.005_l10.01_m0.9.pth'
    save_path = r'C:\Users\Andrii\PycharmProjects\segmentationTraining\models\resnet50_weights\resnet50_brute_1.onnx'

    # set the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('running on device ' + str(device))

    # load the model checkpoint
    print('loading checkpoint:  ' + checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location = device)

    arch = checkpoint['arch']
    num_classes = checkpoint['num_classes']

    # print('checkpoint accuracy: {:.3f}% mean IoU, {:.3f}% accuracy'.format(checkpoint['mean_IoU'], checkpoint['accuracy']))

    # create the model architecture
    print('using model:  ' + arch)
    print('num classes:  ' + str(num_classes))
    # Use this for convinient transition
    model = models.segmentation.__dict__[arch](num_classes=num_classes,
                                               aux_loss=None,
                                               pretrained=False,
                                               export_onnx=True)


    # otherwise use this:
    #model = fcn_resnet34(pretrained=False, progress=True, num_classes = 2, aux_loss=False)
    # load the model weights
    model.load_state_dict(checkpoint['model'])

    model.to(device)
    model.eval()

    print(model)
    print('')

    # create example image data
    resolution = checkpoint['resolution']
    # input = torch.ones((1, 3, resolution[0], resolution[1])).cuda()  #with cuda enabled
    input = torch.ones((1, 3, resolution[0], resolution[1]))  # with cuda disabled
    print('input size:  {:d}x{:d}'.format(resolution[1], resolution[0]))

    # format output model path


    # export the model
    input_names = ["input_0"]
    output_names = ["output_0"]

    print('exporting model to ONNX...')
    torch.onnx.export(model, input, save_path, verbose=True, input_names=input_names, output_names=output_names)
    print('model exported to:  {:s}'.format(save_path))


if __name__ == '__main__':
    # import torchvision
    # model = torchvision.models.segmentation.__dict__['fcn_resnet50'](num_classes=2)
    # checkpoint = torch.load(r'C:\Users\Andrii\PycharmProjects\segmentationTraining\models\resnet50_weights\resnet50_L1_ep18.pth',
    #                         map_location='cpu')
    # model.load_state_dict(checkpoint['model'])
    # model_save_path = r'C:\Users\Andrii\PycharmProjects\segmentationTraining\models\resnet50_weights'
    # torch.save({'model': model.state_dict(),
    #             'num_classes': 2,
    #             'resolution': (320, 320),
    #             'arch': 'fcn_resnet50'}, f'{model_save_path}/resnet50_L1_ep18_correct.pth',
    #            )

    main()


