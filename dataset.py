from torch.utils.data import Dataset
import os
import torch
from PIL import Image
import torchvision.transforms as trans
import numpy as np

class ImgRadarDataset(Dataset):
    def __init__(self, root_dir, transform=None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform if transform is not None else trans.Compose([
            trans.Resize((224, 224)),
            trans.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.05
            ),
            trans.RandomApply([ trans.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5)) ], 0.5) ,
            trans.ToTensor(),
            trans.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


        self.img_dir = os.path.join(self.root_dir, "images")
        self.radar_dir = os.path.join(self.root_dir, "radars")
        self.label_dir = os.path.join(self.root_dir, "labels")

        self.img_pathes = sorted([
            os.path.join(self.img_dir, f) for f in os.listdir(self.img_dir)
        ])
        self.radar_pathes = sorted([
            os.path.join(self.radar_dir, f) for f in os.listdir(self.radar_dir)
        ])
        self.label_pathes = sorted([
            os.path.join(self.label_dir, f) for f in os.listdir(self.label_dir)
        ])
        self.len = len(self.img_pathes)
        assert len(self.radar_pathes) == len(self.label_pathes) == self.len


    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img_path = self.img_pathes[index]
        radar_path = self.radar_pathes[index]
        label_path = self.label_pathes[index]

        img = Image.open(img_path).convert("RGB")
        radar = np.load(radar_path)
        label = np.load(label_path)

        img = self.transform(img)
        radar = torch.from_numpy(radar).unsqueeze(0) # [1, 360]
        label = torch.from_numpy(label)

        return img, radar, label
