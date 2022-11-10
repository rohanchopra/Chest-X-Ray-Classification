import os
import torchvision.transforms as T
from torchvision import datasets
import torch.utils.data as td


def load_data(path, batch_size, input_size, norm_arr, 
              num_workers=0):
    
    transform_dict = {"train": 
                          T.Compose([
                            #T.ToPILImage(),
                            T.Resize(size=input_size),

                            T.RandomHorizontalFlip(),
                            #T.ColorJitter(contrast=0.5),
                            T.RandomAdjustSharpness(2),
                            T.RandomAutocontrast(),

                            T.ToTensor(),
                            T.Normalize(*norm_arr)]),
                        "test_val": 
                          T.Compose([
                            #T.ToPILImage(),
                            T.Resize(size=input_size),
                            T.ToTensor(),
                            T.Normalize(*norm_arr)])
                        }
    
    train_dataset = datasets.ImageFolder(root=os.path.join(path, "train"),
                                         transform=transform_dict["train"])
    val_dataset = datasets.ImageFolder(root=os.path.join(path, "val"),
                                       transform=transform_dict["test_val"])
    test_dataset = datasets.ImageFolder(root=os.path.join(path, "test"),
                                        transform=transform_dict["test_val"])
    
    data_loader_train = td.DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      drop_last=False,
                                      num_workers=num_workers,
                                      pin_memory=True)
    data_loader_val = td.DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=num_workers,
                                    pin_memory=True)
    data_loader_test = td.DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=num_workers,
                                     pin_memory=True)

    return {'train': data_loader_train,
            'val':   data_loader_val,
            'test':  data_loader_test}