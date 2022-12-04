import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class MultiLabelDataset(Dataset):
    def __init__(self, root, dataframe, transform = None):
        self.dataframe = dataframe
        self.root = root
        self.transform = transform
        self.file_names = dataframe.index
        self.labels = dataframe.values.tolist()
        self.classes = list(dataframe.columns)
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, self.file_names[index]))
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float)