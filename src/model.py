from utils import set_requires_grad
from torchvision import models
import torch.nn as nn


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=None):
    '''
    Initialize models for training.
    '''
    model_ft = None
    input_size = 0

    if model_name == "resnet50":
        """ 
            Returns a Resnet 50 model.
        """
        model_ft = models.resnet50(weights=use_pretrained)
        set_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 232
        
    elif model_name == "resnet34":
        """ 
            Returns a Resnet 34 model.
        """
        model_ft = models.resnet34(weights=use_pretrained)
        set_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 256

    elif model_name == "inceptionv3":
        """ 
            Returns an Inception v3 model.
        """
        model_ft = models.inception_v3(weights=use_pretrained)
        set_requires_grad(model_ft, feature_extract)
        # Auxilary network.
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Primary network.
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299
        
    elif model_name == "vgg16":
        """ 
            Returns a VGG 16 model with batch normalization.
        """
        model_ft = models.vgg16_bn(weights=use_pretrained)
        set_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 256

    elif model_name == "mobile_net_v3_large":
        """ 
            returns a mobile net V3 large model.
        """
        model_ft = models.mobilenet_v3_large(weights=use_pretrained)
        set_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[3].in_features
        model_ft.classifier[3] = nn.Linear(num_ftrs, num_classes, bias=True)
        input_size = 232

    elif model_name == "efficient_net_b1":
        """ 
            Returns an efficient net b1 model.
        """
        model_ft = models.efficientnet_b1(weights=use_pretrained)
        set_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes, bias=True)
        input_size = 255

    elif model_name == "efficient_net_b0":
        """ 
            Returns an efficient net b0 model.
        """
        model_ft = models.efficientnet_b0(weights=use_pretrained)
        set_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes, bias=True)
        input_size = 256

    else:
        print("Unavailable model selected.")

    return model_ft, input_size