from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import ReLU
from torch.nn import BatchNorm2d
from torch.nn import Sequential


class DBP(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Initial layer
        self.conv1 = self.initial_layer(in_channels, out_channels = 64, kernel_size = 3, stride= 1, padding = 1)

        # 15 layer, all equals in dimensions (we need to define each so they have different weights)
        self.conv2 = self.conv_block(in_channels = 64, out_channels= 64, kernel_size= 3, stride=1, padding=1)
        self.conv3 = self.conv_block(in_channels = 64, out_channels= 64, kernel_size= 3, stride=1, padding=1)
        self.conv4 = self.conv_block(in_channels = 64, out_channels= 64, kernel_size= 3, stride=1, padding=1)
        self.conv5 = self.conv_block(in_channels = 64, out_channels= 64, kernel_size= 3, stride=1, padding=1)
        self.conv6 = self.conv_block(in_channels = 64, out_channels= 64, kernel_size= 3, stride=1, padding=1)
        self.conv7 = self.conv_block(in_channels = 64, out_channels= 64, kernel_size= 3, stride=1, padding=1)
        self.conv8 = self.conv_block(in_channels = 64, out_channels= 64, kernel_size= 3, stride=1, padding=1)
        self.conv9 = self.conv_block(in_channels = 64, out_channels= 64, kernel_size= 3, stride=1, padding=1)
        self.conv10 = self.conv_block(in_channels = 64, out_channels= 64, kernel_size= 3, stride=1, padding=1)
        self.conv11 = self.conv_block(in_channels = 64, out_channels= 64, kernel_size= 3, stride=1, padding=1)
        self.conv12 = self.conv_block(in_channels = 64, out_channels= 64, kernel_size= 3, stride=1, padding=1)
        self.conv13 = self.conv_block(in_channels = 64, out_channels= 64, kernel_size= 3, stride=1, padding=1)
        self.conv14 = self.conv_block(in_channels = 64, out_channels= 64, kernel_size= 3, stride=1, padding=1)
        self.conv15 = self.conv_block(in_channels = 64, out_channels= 64, kernel_size= 3, stride=1, padding=1)
        self.conv16 = self.conv_block(in_channels = 64, out_channels= 64, kernel_size= 3, stride=1, padding=1)
        
        #Final layer
        self.final = self.final_layer(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)


    def initial_layer(self, in_channels, out_channels, kernel_size, stride, padding):
       initial = Sequential(
                    Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                    ReLU(inplace=True))
       return initial

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
       convolution = Sequential(
                    Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                    BatchNorm2d(out_channels),
                    ReLU(inplace=True))
       return convolution


    def final_layer(self, in_channels, out_channels, kernel_size, stride, padding):
       final = Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
       return final


    def forward(self, x):
        # initial part
        conv1 = self.conv1(x)

        # middle part
        middle2 = self.conv2(conv1)
        middle3 = self.conv3(middle2)
        middle4 = self.conv4(middle3)
        middle5 = self.conv5(middle4)
        middle6 = self.conv6(middle5)
        middle7 = self.conv7(middle6)
        middle8 = self.conv8(middle7)
        middle9 = self.conv9(middle8)
        middle10 = self.conv10(middle9)
        middle11 = self.conv11(middle10)
        middle12 = self.conv12(middle11)
        middle13 = self.conv13(middle12)
        middle14 = self.conv14(middle13)
        middle15 = self.conv15(middle14)
        middle16 = self.conv16(middle15)
        
        #final part
        final_layer = self.final(middle16)

        return final_layer
