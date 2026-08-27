import torch
from model import AutoNet
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as trans
import os
import numpy as np
from PIL import Image


class InfDataset(Dataset):
    """Inference-only dataset. Loads images + radars, no labels required."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or trans.Compose([
            trans.Resize((224, 224)),
            trans.ToTensor(),
            trans.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        self.img_dir = os.path.join(root_dir, "images")
        self.radar_dir = os.path.join(root_dir, "radars")

        self.img_paths = sorted(
            os.path.join(self.img_dir, f) for f in os.listdir(self.img_dir)
        )
        self.radar_paths = sorted(
            os.path.join(self.radar_dir, f) for f in os.listdir(self.radar_dir)
        )
        self.len = len(self.img_paths)
        assert len(self.radar_paths) == self.len, \
            f"images ({self.len}) != radars ({len(self.radar_paths)})"

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        radar_path = self.radar_paths[index]

        img = Image.open(img_path).convert("RGB")
        radar = np.load(radar_path)

        img_tensor = self.transform(img)                     # [3, 224, 224]
        radar_tensor = torch.from_numpy(radar).unsqueeze(0)  # [1, 360]

        return img_tensor, radar_tensor, img_path


def load_model(model_path, device="cpu", radar_length=360):
    """Load AutoNet weights and set to eval mode."""
    model = AutoNet(radar_length)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_inference(model, dataloader, device="cpu"):
    """Run inference over a DataLoader, return list of {path, pred}."""
    results = []
    model.eval()
    with torch.no_grad():
        for img, radar, paths in dataloader:
            img = img.to(device)
            radar = radar.to(device)
            outputs = model((img, radar))  # [B, 2]
            for i in range(len(outputs)):
                results.append({
                    "path": paths[i] if isinstance(paths, (list, tuple))
                           else paths[i].decode("utf-8"),
                    "pred": outputs[i].cpu().numpy(),  # shape (2,)
                })
    return results
