import torch
import torch.nn as nn

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
            #conv2의 default stride=1, padding=0임을 잊지말자
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class Residual_block(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(Residual_block, self).__init__()

        self.conv1 = BasicConv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv2 = BasicConv2d(in_channels // 2, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        return out
    
class Darknet53(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(Darknet53, self).__init__()

        self.stem = nn.Sequential(
            BasicConv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 64, kernel_size=3, stride=2, padding=1)
        )
        
        self.res_conv_1 = self._make_layer(64, 1)

        self.conv_2_in = BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.res_conv_2 = self._make_layer(128, 2)

        self.conv_3_in = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.res_conv_3 = self._make_layer(256, 8)

        self.conv_4_in = BasicConv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.res_conv_4 = self._make_layer(512, 8)

        self.conv_5_in = BasicConv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.res_conv_5 = self._make_layer(1024, 4)

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, num_classes)
        )

    def _make_layer(self, in_channels, num_blocks):
        layers = []
        
        for _ in range(num_blocks):
            layers.append(Residual_block(in_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.res_conv_1(x)
        x = self.conv_2_in(x)
        x = self.res_conv_2(x)
        x = self.conv_3_in(x)
        x = self.res_conv_3(x)
        x = self.conv_4_in(x)
        x = self.res_conv_4(x)
        x = self.conv_5_in(x)
        x = self.res_conv_5(x)
        x = self.fc(x)

        return x


def debug(model, input_size):
    dummy_input = torch.randn(1, *input_size)
    output = model(dummy_input)
    print("Input size:", input_size)
    print("Output size:", output.size())
    
if __name__ == '__main__':
    # 예시 호출
    debug(Darknet53, input_size=(3, 256, 256))
    debug(Darknet53, input_size=(3, 416, 416))