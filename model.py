import torch.nn as nn
import torch
from torchvision.models import resnet
import torch.nn.functional as F
import numpy as np

class AutoNet(nn.Module):
    def __init__(self, radar_length=360):
        super().__init__()
        self.radar_length = radar_length

        self.img_net = resnet.resnet18()
        self.radar_net = Radar1dEncoder(radar_length=self.radar_length)
        self.img_net = nn.Sequential(
            *list(self.img_net.children())[:-1],
            nn.Flatten()
        )

        self.out = nn.Sequential(
            nn.Linear(512 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )


    def forward(self, input):
        img, radar = input
        img = self.img_net(img)
        radar = self.radar_net(radar)

        fusion = torch.cat([img, radar], dim=1)
        result = self.out(fusion)
        return result



class Radar1dEncoder(nn.Module):
    def __init__(self, radar_length=360) -> None:
        super().__init__()
        self.radar_length = radar_length

        angle = torch.linspace(0, 2 * np.pi, self.radar_length)
        self.register_buffer(
                'sin_cos_pos_encode',
                torch.stack(
                    [torch.sin(angle), torch.cos(angle)],
                    dim = 0
                    )
                ) # [2, 360]

        self.radar_encode = nn.Sequential(
            nn.CircularPad1d((3, 3)),  # [B.forard, 3, 366]
            nn.Conv1d(3, 64, kernel_size=7, stride=2),
            # [B, 64, 180] (366 - 7) / 2 + 1
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.CircularPad1d((2, 2)), # [B, 64, 184]
            nn.Conv1d(64, 128, kernel_size=5, stride=2),
            # [B, 128, 90] (184 - 5) / 2
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.CircularPad1d((1, 1)), # [B, 128, 92]
            nn.Conv1d(128, 256, kernel_size=3, stride=2),
            # [B, 256, 45] (92 - 3) / 2 + 1
            nn.BatchNorm1d(256),
            nn.ReLU(),
            )

        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, radar_dist):
        sin_cos = self.sin_cos_pos_encode.unsqueeze(0).repeat(
            radar_dist.size(0), 1, 1
        ) # [B, 2, 360]
        x = torch.cat([radar_dist, sin_cos], dim=1) # [B, 3, 360]
        x = self.radar_encode(x)
        x = self.gap(x).squeeze(-1) # [B, 256]
        return x

if __name__ == "__main__":
    radar_length = 360
    radar_encoder = Radar1dEncoder(radar_length)
    print(radar_encoder)
    radar_dist = torch.rand(2, 1, radar_length) # [B, 1, 360]
    output = radar_encoder(radar_dist)
    print(output.shape) # [B, 256]

    auto_net = AutoNet(radar_length)
    print(auto_net)
    img = torch.rand(2, 3, 640, 640)
    input = (img, radar_dist)
    output = auto_net(input)
    print(output.shape)
