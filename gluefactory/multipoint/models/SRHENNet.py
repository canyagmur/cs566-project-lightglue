<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class HNet(nn.Module):
    def __init__(self):
        super(HNet, self).__init__()
        # Backbone
        base_net = models.resnet34(pretrained=True)
        #base_net.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        base_net.fc = nn.Linear(512, 8)
        base_net.layer4[0].conv1 = nn.Conv2d(1024, 512, 3, 2, 1, bias=False)
        base_net.layer4[0].downsample[0] = nn.Conv2d(1024, 512, 1, 2, bias=False)
        # Feature network
        # self.fnet = nn.Sequential(
        #     base_net.conv1,
        #     base_net.bn1,
        #     base_net.relu,
        #     base_net.maxpool,
        #     base_net.layer1,
        #     base_net.layer2
        # )
        
        # Regression network
        self.rnet = nn.Sequential(
            base_net.layer4,
            base_net.avgpool,
        )
        self.fc = base_net.fc
        del base_net
    
    def forward(self, x1, x2):
        #x1 = self.fnet(x1)
        #x2 = self.fnet(x2)
        x = self._cost_volume(x1, x2)
        x = self.rnet(x)
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
        cv = cv.reshape(N, H*W, H, W)
        return cv
=======
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class HNet(nn.Module):
    def __init__(self):
        super(HNet, self).__init__()
        # Backbone
        base_net = models.resnet34(pretrained=True)
        #base_net.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        base_net.fc = nn.Linear(512, 8)
        base_net.layer4[0].conv1 = nn.Conv2d(1024, 512, 3, 2, 1, bias=False)
        base_net.layer4[0].downsample[0] = nn.Conv2d(1024, 512, 1, 2, bias=False)
        # Feature network
        # self.fnet = nn.Sequential(
        #     base_net.conv1,
        #     base_net.bn1,
        #     base_net.relu,
        #     base_net.maxpool,
        #     base_net.layer1,
        #     base_net.layer2
        # )
        
        # Regression network
        self.rnet = nn.Sequential(
            base_net.layer4,
            base_net.avgpool,
        )
        self.fc = base_net.fc
        del base_net
    
    def forward(self, x1, x2):
        #x1 = self.fnet(x1)
        #x2 = self.fnet(x2)
        x = self._cost_volume(x1, x2)
        x = self.rnet(x)
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
        cv = cv.reshape(N, H*W, H, W)
        return cv
>>>>>>> 1e6332902b2e67c57ce2da44844b926bca185d37
