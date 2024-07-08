import torch
import torch.nn as nn
import torch.nn.functional as F

from darknet53_FPN import *

# FPN모델은 3개의 채널 리스트를 인자값으로 받아야함
features_shape = [256, 512, 1024]

class FPN(nn.Module):
    def __init__(self, channels_list, num_repeats=2):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(channels, channels//2, kernel_size=1)
            for channels in channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            self._make_fpn_layers(channels, num_repeats)
            for channels in channels_list
        ])
        self.tail_convs = nn.ModuleList([
            BasicConv2d(channels // 2, channels, kernel_size=3, padding=1)
            for channels in channels_list
        ])
        self.merge_convs = nn.ModuleList([
            nn.Conv2d(channels_list[i] + channels_list[i+1]//2, channels_list[i]//2, kernel_size=1)
            for i in range(len(channels_list) - 1)
        ]) # channels_list[i+1]//2 연산 수행 후 channels_list[i]랑 덧셈되는 것

    def _make_fpn_layers(self, channels, num_repeats):
        layers = []
        for _ in range(num_repeats):
            layers.append(BasicConv2d(channels // 2, channels, kernel_size=3, padding=1))
            layers.append(BasicConv2d(channels, channels // 2, kernel_size=1))
        return nn.Sequential(*layers)

    def forward(self, *features):
        # 가변 위치인자 '*args'는 튜플로 처리되기에 리스트로 변환시킨다.
        features = list(features)
        # 첫번재 병렬 레이어 lateral_convs를 적용해 리스트 Feature Map : lateral_features 생성
        lateral_features = [lateral_conv(f) for lateral_conv, f in zip(self.lateral_convs, features)]
        # 두번째 병렬 레이어 fpn_convs를 적용해 리스트 fpn_features 피처맵 생성
        fpn_features = [fpn_conv(f) for fpn_conv, f in zip(self.fpn_convs, lateral_features)]

        # FPN의 Top-down pathway(상향경로) and aggregation(통합) 코드
        for i in range(len(fpn_features)-1, 0, -1): 
            # 높은 레벨의 Featue를 낮은 레벨의 Featuer와 합성하기 위해 H, W를 Upsample
            upsampled = F.interpolate(fpn_features[i], scale_factor=2, mode='nearest')
            # 업 샘플된 높은 레벨의 Feature을 낮은 레벨의 Featuer와 Concat
            features[i-1] = torch.cat((features[i-1], upsampled), 1)
            # Concat한 신규 Feature을 차원축소(merge_convs) 하여 첫번째 병렬 레이어 업데이트
            lateral_features[i-1] = self.merge_convs[i-1](features[i-1])
            # 업데이트 된 첫번째 병렬 레이어 정보를 기반으로 두번째 병렬 레이어 업데이트
            fpn_features[i-1] = self.fpn_convs[i-1](lateral_features[i-1])

        # 가장 마지막 3번째 병렬 레이어에 Top-down pathway and aggregation이 적용된 Feature를 적용
        neck_out = [tail_conv(fpn_feature) for tail_conv, fpn_feature in zip(self.tail_convs, fpn_features)]

        return neck_out
    

class YOLOHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(YOLOHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, 3 * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

class YOLOv3(nn.Module):
    def __init__(self, backbone, fpn, num_classes=80):
        super(YOLOv3, self).__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.num_classes = num_classes

        self.heads = nn.ModuleList([
            YOLOHead(in_channels, num_classes) for in_channels in features_shape
        ])

    def forward(self, x):
        features = self.backbone(x)
        fpn_features = self.fpn(*features)
        outputs = [head(fpn_feature) for head, fpn_feature in zip(self.heads, fpn_features)]
        return outputs
    

def debug(model, input_size):
    dummy_input = torch.randn(1, *input_size)
    outputs = model(dummy_input)
    print("Input size:", input_size)
    print("Output type:", type(outputs))
    for idx, output in enumerate(outputs):
        print(f"Output {idx} shape: {output.shape}")


if __name__ == '__main__':
    backbone = Darknet53(pretrained=True)
    fpn = FPN(channels_list=features_shape)
    yolov3 = YOLOv3(backbone, fpn, num_classes=80)
    debug(yolov3, input_size=(3, 416, 416))