import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class HNetLighter(nn.Module):
    def __init__(self):
        super(HNetLighter, self).__init__()

        # Backbone
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            #nn.Linear(128 * 16 * 16, 1024),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(64, 8)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
    
    def forward(self, x1, x2):
        x1 = self.layer1(x1)
        x2 = self.layer1(x2)
        x = self._cost_volume(x1, x2)
        x = self.avg_pool(x)
        B,C,H,W = x.shape
        x = x.view(B,C)
        x = self.fc(x)
        return x
    
    @staticmethod
    def _cost_volume(x1, x2):
        N, C, H, W = x1.shape
        x1 = F.normalize(x1)
        x2 = F.normalize(x2)
        x1 = x1.reshape(N, C, H*W)
        x2 = x2.reshape(N, C, H*W)
        cv = torch.bmm(x1.transpose(1, 2), x2)
        cv = cv.reshape(N, H*W, H, W) #cv.reshape(N, H*W//4, H*2, W*2) #cv.reshape(N, H*W, H, W) #cv.reshape(N, H*W//4, H*2, W*2) #LOOK HERE
        return cv
