import os
import torch


def set_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
            
def save_model(model, path, name):
    torch.save(model, os.path.join(path, name))
    return True