import torch
import torch.nn as nn
from model.AGSCNet import *
from model.High_Frequency_Module import HighFrequencyModule

class Heatmap(nn.Module):
    def __init__(self):
        super(Heatmap, self).__init__()
        self.model = AGSCNet(n_channels=3, n_classes=1)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device=device)
        # print(self.net)
        self.model.load_state_dict(torch.load('Backbone_CLAM_ASPP_CCNet-SFusion_out1_Dice_out2_BCE_BestmIoU_epoch_59_mIoU_89.783295.pth', map_location=device))
        self.sigmoid = nn.Sigmoid()
        self.edge = HighFrequencyModule(input_channel=64, the_filter='Isotropic_Sobel')
    def forward(self, x):
        x = self.model.inc(x)
        x1 = x

        x = self.model.down1(x)
        x2 = x

        x = self.model.down2(x)
        x3 = x

        x = self.model.down3(x)
        x4 = x

        x = self.model.down4(x)

        context = self.model.SAM(x)
        aspp = self.model.aspp(x)
        content = self.model.CAM(x, aspp)

        alpha = self.model.sigmoid(self.model.gamma)
        CCNet = alpha * context + (1 - alpha) * content

        x = self.model.up1(CCNet, x4)

        x = self.model.up2(x, x3)

        x = self.model.up3(x, x2)

        x = self.model.up4(x, x1)

        x = self.model.outConv1(x)
        out1 = self.model.sigmoid(x)

        out = self.model.sigmoid(x)

        return [aspp, context, content, CCNet]

if __name__ == '__main__':
    net = AGSCNet(n_channels=3, n_classes=1)
    print(net)