""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import logging



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.in_planes =64
        self.segmentation_channels = 32

        
        self.inc = DoubleConv(n_channels, 64)  
        self.down1 = self._make_layer(BasicBlock, 128, 3, stride=2)
        self.down2 = self._make_layer(BasicBlock, 256, 4, stride=2)
        self.down3 = self._make_layer(BasicBlock, 512, 6, stride=2)
        self.down4 = self._make_layer(BasicBlock, 512, 3, stride=2)
        
        self.up1 = resUp(1024, 512 // 2, 2)
        self.up2 = resUp(512, 256 // 2 , 2)
        self.up3 = resUp(256, 128 // 2, 2)
        self.up4 = resUp(128, 64, 2)
        
        #self.up3 = resUp_v2(384, 128 // 2, 2)
        #self.up4 = resUp_v2(192, 64, 2)
        
        #self.dup1=DUpsampling(inplanes = 512, scale = 8, num_class = 128, pad=0)
        #self.dup2=DUpsampling(inplanes = 256, scale = 8, num_class = 64, pad=0)
        
        self.cbam1=cbam(64)
        self.cbam2=cbam(128)
        self.cbam3=cbam(256)
        self.cbam4=cbam(512)
        self.cbam5=cbam(512)
        self.cbam6=cbam(256)
        self.cbam7=cbam(128)
        self.cbam8=cbam(64)
        
        #A2FPN部分
        self.s5 = SegmentationBlock(256, self.segmentation_channels, n_upsamples=3)
        self.s4 = SegmentationBlock(128, self.segmentation_channels, n_upsamples=2)
        self.s3 = SegmentationBlock(64, self.segmentation_channels, n_upsamples=1)
        self.s2 = SegmentationBlock(64, self.segmentation_channels, n_upsamples=0)

        self.attention = SE(self.segmentation_channels * 4)
        
        
        #resup和attentiondown,失败了
        #self.down1 = self._make_layer(BasicBlock, 128, 1, stride=2)
        #self.down2 = self._make_layer(BasicBlock, 256, 1, stride=2)
        #self.down3 = self._make_layer(BasicBlock, 512, 1, stride=2)
        #self.down4 = self._make_layer(BasicBlock, 512, 1, stride=2)
        
        #self.up1 = MyUp(image_size=(40,80), patch_size=20, dim=512, heads=8, in_channels=512, out_channels=512)
        #self.up2 = MyUp(image_size=(80,160), patch_size=20, dim=512, heads=8, in_channels=512, out_channels=256)
        #self.up3 = MyUp(image_size=(160,320), patch_size=20, dim=512, heads=8, in_channels=256, out_channels=128)
        #self.up4 = MyUp(image_size=(320,640), patch_size=20, dim=512, heads=8, in_channels=128, out_channels=64)        

        #原来的up和down
        #self.down1 = Down(64, 128)
        #self.down2 = Down(128, 256)
        #self.down3 = Down(256, 512)
        #factor = 2 if bilinear else 1
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        #self.up2 = Up(512, 256 // factor, bilinear)
        #self.up3 = Up(256, 128 // factor, bilinear)
        #self.up4 = Up(128, 64, bilinear)
        #self.outc = OutConv(64, n_classes)
        self.outc = OutConv(self.segmentation_channels * 4, n_classes)

    def forward(self, x):
        # logging.info("input shape:" + str(x.shape))
        # logging.info("# of values >= 1 in x:" + str(torch.sum(x >= 1.0)))
        x1 = self.inc(x)        
        x1 = self.cbam1(x1)
        x2 = self.down1(x1)
        x2 = self.cbam2(x2)
        #x3 = self.cbam2(x2)
        x3 = self.down2(x2)
        x3 = self.cbam3(x3)
        #x4 = self.cbam3(x3)
        x4 = self.down3(x3)
        x4 = self.cbam4(x4)
        #x5 = self.cbam4(x4)
        x5 = self.down4(x4)
        x5 = self.cbam5(x5)
        # logging.info("# of values >= 1 in x5:" + str(torch.sum(x5 >= 1.0)))
        
        #shortcut_x5=self.dup1(x5)
        x6 = self.up1(x5 , x4)
        
        x7 = self.cbam6(x6)
        #shortcut_x6=self.dup2(x6)
        x7 = self.up2(x7, x3)
        
        x8 = self.cbam7(x7)
        #x = self.up3(x, x2)
        x8 = self.up3(x8, x2)
        
        x9 = self.cbam8(x8)
        #x = self.up4(x, x1)
        x9 = self.up4(x9, x1)        
        
        
        #A2FPN,这里记得改
        s5 = self.s5(x6)
        s4 = self.s4(x7)
        s3 = self.s3(x8)
        s2 = self.s2(x9)

        x = self.attention(torch.cat([s5, s4, s3, s2], dim = 1))
        
        
        # logging.info("# of values >= 1 after up4:" + str(torch.sum(x >= 1.0)))
        out = self.outc(x)
        # logging.info("output shape:" + str(logits.shape))
        # logging.info("# of values >= 1 in logit:" + str(torch.sum(logits >= 1.0)))

        # (N1, C1, H1, W1) of input x,  (N2, C2, H2, W2) of output(aka loggits),
        # assert H1 == H2, W1 == W2
        return out

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)
    
