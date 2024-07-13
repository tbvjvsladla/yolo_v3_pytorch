import torch
import torch.nn as nn
import numpy as np

def parse_cfg(cfg_file):
    with open(cfg_file, 'r') as file:
        lines = file.read().split('\n')
        lines = [x for x in lines if x and not x.startswith('#')]
        lines = [x.rstrip().lstrip() for x in lines]
    
    block = {}
    blocks = []

    for line in lines:
        if line.startswith('['):
            if block:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

def convert_anchor(cfg_file):
    blocks = parse_cfg(cfg_file)
    net_info = blocks[0]
    width = int(net_info['width'])
    height = int(net_info['height'])
    
    anchors = []
    
    for block in blocks:
        if block['type'] == 'yolo':
            mask = [int(m) for m in block['mask'].split(',')]
            anchor_list = block['anchors'].split(',')
            anchor_list = [int(a) for a in anchor_list]
            anchor_pairs = [(anchor_list[i], anchor_list[i + 1]) for i in range(0, len(anchor_list), 2)]
            anchors.append([anchor_pairs[m] for m in mask])
    
    # 작은 객체, 중간 객체, 큰 객체 용으로 정렬
    small_objects = [(w / width, h / height) for w, h in anchors[2]]
    medium_objects = [(w / width, h / height) for w, h in anchors[1]]
    large_objects = [(w / width, h / height) for w, h in anchors[0]]
    
    anchor_box_list = [small_objects, medium_objects, large_objects]

    return torch.tensor(anchor_box_list)


class EmptyLayer(nn.Module):
    def forward(self, x):
        return x

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

    def forward(self, x, input_dim, num_classes, confidence, nms_conf=0.4):
        # Write results will be implemented here
        pass

class BaseDarknet(nn.Module):
    def __init__(self, blocks):
        super(BaseDarknet, self).__init__()
        self.blocks = blocks
        self.net_info, self.module_list = self.create_modules(blocks)

    def create_modules(self, blocks):
        net_info = blocks[0]
        module_list = nn.ModuleList()
        prev_filters = 3
        output_filters = []

        for index, x in enumerate(blocks[1:]):
            module = nn.Sequential()

            if x['type'] == 'convolutional':
                activation = x['activation']
                try:
                    batch_normalize = int(x['batch_normalize'])
                    bias = False
                except:
                    batch_normalize = 0
                    bias = True

                filters = int(x['filters'])
                padding = int(x['pad'])
                kernel_size = int(x['size'])
                stride = int(x['stride'])

                if padding:
                    pad = (kernel_size - 1) // 2
                else:
                    pad = 0

                conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
                module.add_module(f"conv_{index}", conv)

                if batch_normalize:
                    bn = nn.BatchNorm2d(filters)
                    module.add_module(f"batch_norm_{index}", bn)

                if activation == "leaky":
                    activn = nn.LeakyReLU(0.1, inplace=True)
                    module.add_module(f"leaky_{index}", activn)

            elif x['type'] == 'upsample':
                stride = int(x['stride'])
                upsample = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
                module.add_module(f"upsample_{index}", upsample)

            elif x['type'] == 'route':
                x['layers'] = x['layers'].split(',')
                start = int(x['layers'][0])
                try:
                    end = int(x['layers'][1])
                except:
                    end = 0
                if start > 0:
                    start = start - index
                if end > 0:
                    end = end - index
                route = EmptyLayer()
                module.add_module(f"route_{index}", route)
                if end < 0:
                    filters = output_filters[index + start] + output_filters[index + end]
                else:
                    filters = output_filters[index + start]

            elif x['type'] == 'shortcut':
                shortcut = EmptyLayer()
                module.add_module(f"shortcut_{index}", shortcut)

            elif x['type'] == 'maxpool':
                stride = int(x['stride'])
                size = int(x['size'])
                maxpool = nn.MaxPool2d(size, stride)
                module.add_module(f"maxpool_{index}", maxpool)

            elif x['type'] == 'yolo':
                mask = x['mask'].split(',')
                mask = [int(m) for m in mask]

                anchors = x['anchors'].split(',')
                anchors = [int(a) for a in anchors]
                anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in mask]

                detection = DetectionLayer(anchors)
                module.add_module(f"Detection_{index}", detection)

            prev_filters = filters
            output_filters.append(filters)
            module_list.append(module)

        return net_info, module_list

class Darknet_cfg(BaseDarknet):
    def __init__(self, blocks):
        super(Darknet_cfg, self).__init__(blocks)

    def forward(self, x):
        outputs = {}
        for i, module in enumerate(self.module_list):
            module_type = self.blocks[i + 1]['type']

            if module_type in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_type == "route":
                layers = [int(a) for a in self.blocks[i + 1]["layers"].split(',')]
                if layers[0] > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
            elif module_type == "shortcut":
                from_layer = int(self.blocks[i + 1]["from"])
                x = outputs[i - 1] + outputs[i + from_layer]

            outputs[i] = x

        return x

class Yolo_v3_cfg(BaseDarknet):
    def __init__(self, blocks):
        super(Yolo_v3_cfg, self).__init__(blocks)

    def forward(self, x):
        outputs = {}
        yolo_outputs = []  # 여러 yolo 레이어의 출력을 저장할 리스트
        for i, module in enumerate(self.module_list):
            module_type = self.blocks[i + 1]['type']

            if module_type in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_type == "route":
                layers = self.blocks[i + 1]["layers"]
                if isinstance(layers, str):
                    layers = layers.split(',')
                layers = [int(a) for a in layers]
                if layers[0] > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
            elif module_type == "shortcut":
                from_layer = int(self.blocks[i + 1]["from"])
                x = outputs[i - 1] + outputs[i + from_layer]
            elif module_type == 'yolo':
                yolo_outputs.append(x)  # yolo 레이어의 출력을 저장
            outputs[i] = x

            yolo_outputs.reverse() #출력을 작은 -> 중간 -> 큰으로 재배치

        return yolo_outputs  # 모든 yolo 레이어의 출력을 반환

def load_weights(model, weightfile):
    fp = open(weightfile, 'rb')
    header = np.fromfile(fp, dtype=np.int32, count=5)
    model.header = torch.from_numpy(header)
    model.seen = model.header[3]

    weights = np.fromfile(fp, dtype=np.float32)
    fp.close()

    ptr = 0
    for i, module in enumerate(model.module_list):
        module_type = model.blocks[i + 1]["type"]

        if module_type == "convolutional":
            conv_layer = module[0]
            if "batch_normalize" in model.blocks[i + 1]:
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b

            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w

        if ptr >= len(weights):
            break

def debug(model, input_size):
    dummy_input = torch.randn(1, *input_size)
    outputs = model(dummy_input)
    print("Input size:", input_size)
    if isinstance(outputs, list):
        print(f"Outputs data type: {type(outputs)}")
        for i, output in enumerate(outputs):
            print(f"Output {i} size:", output.size())
    else:
        print("Output size:", outputs.size())

if __name__ == '__main__':
    # Darknet53 모델
    cfg_file = "darknet53.cfg"
    weight_file = "darknet53.weights"
    blocks = parse_cfg(cfg_file)
    cfg_model = Darknet_cfg(blocks)
    load_weights(cfg_model, weight_file)
    print("Darknet53 모델 디버깅")
    debug(cfg_model, input_size=(3, 256, 256))

    # YOLOv3 모델
    cfg_file = "yolov3.cfg"
    weight_file = "YOLOv3-416.weights"
    blocks = parse_cfg(cfg_file)
    yolo_model = Yolo_v3_cfg(blocks)
    load_weights(yolo_model, weight_file)
    print("YOLOv3 모델 디버깅")
    debug(yolo_model, input_size=(3, 416, 416))

    # Anchor 변환 테스트
    cfg_anchor_box_list = convert_anchor(yolo_model)
    print("Anchor Box List:", cfg_anchor_box_list)