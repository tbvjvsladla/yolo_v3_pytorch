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

class EmptyLayer(nn.Module):
    def forward(self, x):
        return x
    

class Darknet_cfg(nn.Module):
    def __init__(self, blocks):
        super(Darknet_cfg, self).__init__()
        self.blocks = blocks
        self.net_info, self.module_list = self.create_modules(blocks)

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

            prev_filters = filters
            output_filters.append(filters)
            module_list.append(module)

        return net_info, module_list
    

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
            # print(f"Loading weights for layer {i} of type {module_type}")

            if "batch_normalize" in model.blocks[i + 1]:
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()
                # print(f"Batch norm layer: {num_b} biases, {num_b} weights, {num_b} running means, {num_b} running vars")
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
                # print(f"Conv layer: {num_b} biases")
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b

            num_w = conv_layer.weight.numel()
            # print(f"Conv layer: {num_w} weights")
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w

        if ptr >= len(weights):
            print("Finished loading weights")
            break


def debug(model, input_size):
    dummy_input = torch.randn(1, *input_size)
    output = model(dummy_input)
    print("Input size:", input_size)
    print("Output size:", output.size())

if __name__ == '__main__':

    cfg_file = "darknet53.cfg"
    weight_file = "darknet53.weights"

    blocks = parse_cfg(cfg_file)
    cfg_model = Darknet_cfg(blocks)
    load_weights(cfg_model, weight_file)

    debug(cfg_model, input_size=(3, 256, 256))