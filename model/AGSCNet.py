from model.CLAM_Decoder import *
from model.ASPP import AtrousPyramidModule
from model.SelfAttentionModule import SAM
from model.SelfAttentionModule import CAM
from torch.nn import Parameter

class AGSCNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(AGSCNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)

        self.down1 = Down(64, 128)

        self.down2 = Down(128, 256)

        self.down3 = Down(256, 512)

        self.down4 = Down(512, 1024)

        self.aspp = AtrousPyramidModule(in_channel=1024, out_channel=1024, rate=[1, 2, 4, 8])

        self.CAM = CAM(hf_channels=1024, lf_channels=1024)
        self.SAM = SAM(in_dim=1024)
        self.gamma = Parameter(torch.zeros(1))

        self.up1 = Up(1024, 512, 512, bilinear)

        self.up2 = Up(512, 256, 256, bilinear)

        self.up3 = Up(256, 128, 128, bilinear)

        self.up4 = Up(128, 64, 64, bilinear)

        self.outConv1 = OutConv(64, n_classes)



        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.inc(x)
        x1 = x

        x = self.down1(x)
        x2 = x

        x = self.down2(x)
        x3 = x

        x = self.down3(x)
        x4 = x

        x = self.down4(x)

        context = self.SAM(x)
        aspp = self.aspp(x)
        content = self.CAM(context, aspp)

        alpha = self.sigmoid(self.gamma)
        x = alpha * context + (1 - alpha) * content

        x = self.up1(x, x4)

        x = self.up2(x, x3)

        x = self.up3(x, x2)

        x = self.up4(x, x1)

        x = self.outConv1(x)

        out1 = self.sigmoid(x)

        out = self.sigmoid(x)

        return out1, out

if __name__ == '__main__':
    net = AGSCNet(n_channels=3, n_classes=1)
    m = torch.randn(1,3,224,224)
    print(net(m)[2] .shape)
