from .resnet_backbones import *
from torch.hub import load_state_dict_from_url
from torch import nn

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    Examples::
    """

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out

class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier, aux_classifier=None, export_onnx=False):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier
        self.export_onnx = export_onnx

        print('torchvision.models.segmentation.FCN() => configuring model for ' + (
            'ONNX export' if export_onnx else 'training'))

    def forward(self, x):
        input_shape = x.shape[-2:]

        # contract: features is a dict of tensors
        features = self.backbone(x)
        x = features["out"]
        x = self.classifier(x)

        # TensorRT doesn't support bilinear upsample, so when exporting to ONNX,
        # use nearest-neighbor upsampling, and also return a tensor (not an OrderedDict)
        if self.export_onnx:
            print('FCN configured for export to ONNX')
            print('FCN model input size = ' + str(input_shape))
            print('FCN classifier output size = ' + str(x.size()))

            # x = F.interpolate(x, size=(int(input_shape[0]), int(input_shape[1])), mode='nearest')

            print('FCN upsample() output size = ' + str(x.size()))
            print('FCN => returning tensor instead of OrderedDict')
            return x

        # non-ONNX training/eval path
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        result = OrderedDict()
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result

class FCN(_SimpleSegmentationModel):
    """
    Implements a Fully-Convolutional Network for semantic segmentation.
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)

def fcn_resnet34(pretrained=False, progress=True,
                 num_classes=21, aux_loss=None, **kwargs):
    """Constructs a Fully-Convolutional Network model with a ResNet-34 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    print('torchvision.models.segmentation.fcn_resnet34()')

    if pretrained:
        aux_loss = False
    model = _segm_resnet("fcn", "resnet34", num_classes, aux_loss, **kwargs)
    if pretrained:
        arch = 'fcn_resnet34_coco'
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model.load_state_dict(state_dict)
    return model

def _segm_resnet(name, backbone_name, num_classes, aux = False, pretrained_backbone=True, export_onnx=False):

    if backbone_name == "resnet18" or backbone_name == "resnet34":
        replace_stride_with_dilation=[False, False, False]
        inplanes_scale_factor = 4
    else:
        replace_stride_with_dilation=[False, True, True]
        inplanes_scale_factor = 1
    if backbone_name == 'resnet34':
        print('here')
        backbone = resnet34(
            pretrained=pretrained_backbone,
            replace_stride_with_dilation=replace_stride_with_dilation)

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    if aux:
        inplanes = 1024 / inplanes_scale_factor
        aux_classifier = FCNHead(inplanes, num_classes)

    model_map = {
        'fcn': (FCNHead, FCN),
    }

    inplanes = 2048 / inplanes_scale_factor
    classifier = model_map[name][0](int(inplanes), int(num_classes))
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, aux_classifier, export_onnx)
    return model

