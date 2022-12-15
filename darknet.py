from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
from util import * 



def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

def parse_cfg(cfgfile):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """
    
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                        # store the lines in a list
    lines = [x for x in lines if len(x) > 0]               # get read of the empty lines 
    lines = [x for x in lines if x[0] != '#']              # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces
    
    block = {}
    blocks = []
    
    for line in lines:
        if line[0] == "[":               # This marks the start of a new block
            if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)     # add it the blocks list
                block = {}               # re-init the block
            block["type"] = line[1:-1].rstrip()     
        else:
            key,value = line.split("=") 
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    
    return blocks


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
        

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors



def create_modules(blocks):
    net_info = blocks[0]     #Captures the information about the input and pre-processing    
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []
    
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
    
        #check the type of block
        #create a new module for the block
        #append to module_list
        
        #If it's a convolutional layer
        if (x["type"] == "convolutional"):
            #Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
        
            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
        
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
        
            #Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)
        
            #Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
        
            #Check the activation. 
            #It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)
        
            #If it's an upsampling layer
            #We use Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
            module.add_module("upsample_{}".format(index), upsample)
                
        #If it is a route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            #Start  of a route
            start = int(x["layers"][0])
            #end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            #Positive anotation
            if start > 0: 
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]
    
        #shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
            
        #Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
    
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]
    
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)
                              
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        
    return (net_info, module_list)

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
        
    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}   #We cache the outputs for the route layer
        
        write = 0
        for i, module in enumerate(modules):        
            module_type = (module["type"])
            
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
    
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
    
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
    
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
    
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
    
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
                
    
            elif  module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]
    
            elif module_type == 'yolo':        
                anchors = self.module_list[i][0].anchors
                #Get the input dimensions
                inp_dim = int (self.net_info["height"])
        
                #Get the number of classes
                num_classes = int (module["classes"])
        
                #Transform 
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:              #if no collector has been intialised. 
                    detections = x
                    write = 1
        
                else:       
                    detections = torch.cat((detections, x), 1)
        
            outputs[i] = x
        
        return detections

    def load_weights(self, weightfile):
            #Open the weights file
        fp = open(weightfile, "rb")
    
        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]   
        
        weights = np.fromfile(fp, dtype = np.float32)
        
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
    
            #If module_type is convolutional load weights
            #Otherwise ignore.
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
            
                conv = model[0]
                
                
                if (batch_normalize):
                    bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
        
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

# model = Darknet("config/yolov3.cfg")
# model.load_weights("yolov3.weights")
# inp = get_test_input()
# pred = model(inp, torch.cuda.is_available())
# print (pred.shape)












#####################################################################################################################################################################################################################################################################################################################################################################

# blocks = parse_cfg("config/yolov3.cfg")
# print(create_modules(blocks))

# ({'type': 'net', 'batch': '64', 'subdivisions': '16', 'width': '608', 'height': '608', 'channels': '3', 'momentum': '0.9', 'decay': '0.0005', 'angle': '0', 'saturation': '1.5', 'exposure': '1.5', 'hue': '.1', 'learning_rate': '0.001', 'burn_in': '1000', 'max_batches': '500200', 'policy': 'steps', 'steps': '400000,450000', 'scales': '.1,.1'}, ModuleList(
#   (0): Sequential(
#     (conv_0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_0): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_0): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (1): Sequential(
#     (conv_1): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#     (batch_norm_1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_1): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (2): Sequential(
#     (conv_2): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_2): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (3): Sequential(
#     (conv_3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_3): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (4): Sequential(
#     (shortcut_4): EmptyLayer()
#   )
#   (5): Sequential(
#     (conv_5): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#     (batch_norm_5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_5): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (6): Sequential(
#     (conv_6): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_6): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_6): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (7): Sequential(
#     (conv_7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_7): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (8): Sequential(
#     (shortcut_8): EmptyLayer()
#   )
#   (9): Sequential(
#     (conv_9): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_9): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_9): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (10): Sequential(
#     (conv_10): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_10): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_10): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (11): Sequential(
#     (shortcut_11): EmptyLayer()
#   )
#   (12): Sequential(
#     (conv_12): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#     (batch_norm_12): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_12): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (13): Sequential(
#     (conv_13): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_13): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_13): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (14): Sequential(
#     (conv_14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_14): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_14): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (15): Sequential(
#     (shortcut_15): EmptyLayer()
#   )
#   (16): Sequential(
#     (conv_16): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_16): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_16): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (17): Sequential(
#     (conv_17): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_17): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_17): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (18): Sequential(
#     (shortcut_18): EmptyLayer()
#   )
#   (19): Sequential(
#     (conv_19): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_19): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_19): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (20): Sequential(
#     (conv_20): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_20): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_20): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (21): Sequential(
#     (shortcut_21): EmptyLayer()
#   )
#   (22): Sequential(
#     (conv_22): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_22): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_22): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (23): Sequential(
#     (conv_23): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_23): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_23): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (24): Sequential(
#     (shortcut_24): EmptyLayer()
#   )
#   (25): Sequential(
#     (conv_25): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_25): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_25): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (26): Sequential(
#     (conv_26): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_26): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_26): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (27): Sequential(
#     (shortcut_27): EmptyLayer()
#   )
#   (28): Sequential(
#     (conv_28): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_28): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_28): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (29): Sequential(
#     (conv_29): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_29): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_29): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (30): Sequential(
#     (shortcut_30): EmptyLayer()
#   )
#   (31): Sequential(
#     (conv_31): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_31): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_31): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (32): Sequential(
#     (conv_32): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_32): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_32): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (33): Sequential(
#     (shortcut_33): EmptyLayer()
#   )
#   (34): Sequential(
#     (conv_34): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_34): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_34): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (35): Sequential(
#     (conv_35): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_35): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_35): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (36): Sequential(
#     (shortcut_36): EmptyLayer()
#   )
#   (37): Sequential(
#     (conv_37): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#     (batch_norm_37): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_37): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (38): Sequential(
#     (conv_38): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_38): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_38): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (39): Sequential(
#     (conv_39): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_39): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_39): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (40): Sequential(
#     (shortcut_40): EmptyLayer()
#   )
#   (41): Sequential(
#     (conv_41): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_41): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_41): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (42): Sequential(
#     (conv_42): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_42): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_42): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (43): Sequential(
#     (shortcut_43): EmptyLayer()
#   )
#   (44): Sequential(
#     (conv_44): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_44): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_44): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (45): Sequential(
#     (conv_45): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_45): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_45): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (46): Sequential(
#     (shortcut_46): EmptyLayer()
#   )
#   (47): Sequential(
#     (conv_47): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_47): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_47): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (48): Sequential(
#     (conv_48): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_48): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_48): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (49): Sequential(
#     (shortcut_49): EmptyLayer()
#   )
#   (50): Sequential(
#     (conv_50): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_50): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_50): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (51): Sequential(
#     (conv_51): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_51): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_51): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (52): Sequential(
#     (shortcut_52): EmptyLayer()
#   )
#   (53): Sequential(
#     (conv_53): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_53): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_53): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (54): Sequential(
#     (conv_54): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_54): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_54): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (55): Sequential(
#     (shortcut_55): EmptyLayer()
#   )
#   (56): Sequential(
#     (conv_56): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_56): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_56): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (57): Sequential(
#     (conv_57): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_57): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_57): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (58): Sequential(
#     (shortcut_58): EmptyLayer()
#   )
#   (59): Sequential(
#     (conv_59): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_59): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_59): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (60): Sequential(
#     (conv_60): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_60): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_60): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (61): Sequential(
#     (shortcut_61): EmptyLayer()
#   )
#   (62): Sequential(
#     (conv_62): Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#     (batch_norm_62): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_62): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (63): Sequential(
#     (conv_63): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_63): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_63): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (64): Sequential(
#     (conv_64): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_64): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_64): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (65): Sequential(
#     (shortcut_65): EmptyLayer()
#   )
#   (66): Sequential(
#     (conv_66): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_66): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_66): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (67): Sequential(
#     (conv_67): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_67): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_67): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (68): Sequential(
#     (shortcut_68): EmptyLayer()
#   )
#   (69): Sequential(
#     (conv_69): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_69): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_69): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (70): Sequential(
#     (conv_70): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_70): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_70): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (71): Sequential(
#     (shortcut_71): EmptyLayer()
#   )
#   (72): Sequential(
#     (conv_72): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_72): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_72): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (73): Sequential(
#     (conv_73): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_73): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_73): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (74): Sequential(
#     (shortcut_74): EmptyLayer()
#   )
#   (75): Sequential(
#     (conv_75): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_75): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_75): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (76): Sequential(
#     (conv_76): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_76): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_76): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (77): Sequential(
#     (conv_77): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_77): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_77): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (78): Sequential(
#     (conv_78): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_78): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_78): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (79): Sequential(
#     (conv_79): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_79): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_79): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (80): Sequential(
#     (conv_80): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_80): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_80): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (81): Sequential(
#     (conv_81): Conv2d(1024, 255, kernel_size=(1, 1), stride=(1, 1))
#   )
#   (82): Sequential(
#     (Detection_82): DetectionLayer()
#   )
#   (83): Sequential(
#     (route_83): EmptyLayer()
#   )
#   (84): Sequential(
#     (conv_84): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_84): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_84): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (85): Sequential(
#     (upsample_85): Upsample(scale_factor=2.0, mode=bilinear)
#   )
#   (86): Sequential(
#     (route_86): EmptyLayer()
#   )
#   (87): Sequential(
#     (conv_87): Conv2d(768, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_87): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_87): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (88): Sequential(
#     (conv_88): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_88): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_88): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (89): Sequential(
#     (conv_89): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_89): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_89): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (90): Sequential(
#     (conv_90): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_90): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_90): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (91): Sequential(
#     (conv_91): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_91): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_91): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (92): Sequential(
#     (conv_92): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_92): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_92): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (93): Sequential(
#     (conv_93): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
#   )
#   (94): Sequential(
#     (Detection_94): DetectionLayer()
#   )
#   (95): Sequential(
#     (route_95): EmptyLayer()
#   )
#   (96): Sequential(
#     (conv_96): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_96): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_96): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (97): Sequential(
#     (upsample_97): Upsample(scale_factor=2.0, mode=bilinear)
#   )
#   (98): Sequential(
#     (route_98): EmptyLayer()
#   )
#   (99): Sequential(
#     (conv_99): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_99): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_99): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (100): Sequential(
#     (conv_100): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_100): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_100): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (101): Sequential(
#     (conv_101): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_101): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_101): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (102): Sequential(
#     (conv_102): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_102): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_102): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (103): Sequential(
#     (conv_103): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (batch_norm_103): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_103): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (104): Sequential(
#     (conv_104): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (batch_norm_104): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (leaky_104): LeakyReLU(negative_slope=0.1, inplace=True)
#   )
#   (105): Sequential(
#     (conv_105): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
#   )
#   (106): Sequential(
#     (Detection_106): DetectionLayer()
#   )
# ))