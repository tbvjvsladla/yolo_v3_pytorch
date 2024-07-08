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
    def __init__(self, in_channels=3, num_classes=1000, pretrained=False):
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

        if pretrained:
            self.load_weights()

    def _make_layer(self, in_channels, num_blocks):
        layers = []
        
        for _ in range(num_blocks):
            layers.append(Residual_block(in_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        f1 = self.res_conv_1(x)  # 1st level features

        x = self.conv_2_in(f1)
        f2 = self.res_conv_2(x)  # 2nd level features

        x = self.conv_3_in(f2)
        f3 = self.res_conv_3(x)  # 3rd level features

        x = self.conv_4_in(f3)
        f4 = self.res_conv_4(x)  # 4th level features

        x = self.conv_5_in(f4)
        f5 = self.res_conv_5(x)  # 5th level features

        features = [f3, f4, f5]

        return features
    
    def load_weights(self, weight_path='DarkNet53.pth'):
        state_dict = torch.load(weight_path)
        self.load_state_dict(state_dict)


def debug(model, input_size):
    dummy_input = torch.randn(1, *input_size)
    outputs = model(dummy_input)
    print("Input size:", input_size)
    print("Output type:", type(outputs))
    for idx, output in enumerate(outputs):
        print(f"Output {idx} shape: {output.shape}")
    
if __name__ == '__main__':
    # 예시 호출
    debug(Darknet53, input_size=(3, 416, 416))