import os
import torch


def set_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
            
def save_model(model, path, name):
    torch.save(model, os.path.join(path, name))
    return True


def encode_label(label, classes_list): #encoding the classes into a tensor of shape (11) with 0 and 1s.
    target = torch.zeros(len(classes_list))
    for l in label:
        idx = classes_list.index(l)
        target[idx] = 1
    return target


def decode_target(classes, target, threshold=0.5): #decoding the prediction tensors of 0s and 1s into text form
    result = []
    for i, x in enumerate(target):
        if (x >= threshold):
            result.append(classes[i])     
    return ' '.join(result)