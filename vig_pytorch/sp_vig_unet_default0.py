import math
import numpy as np
#import nvsmi
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from gcn_lib import Grapher, act_layer

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

#from skimage.segmentation import slic
from skimage import data

from fast_graph import create_graph


import cupy
#from cuda_slic.slic import slic as cuda_slic
#from mod_cupy_slic import slic as mod_cuda_slic
#from cuda_slic.slic import *
from skimage.segmentation import slic as sk_slic

from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

from torch.nn import Linear, ReLU, Dropout

from torch_geometric.data import Data
from torch_geometric.data import Batch
import torch_geometric.nn as pygnn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import norm
from torch_geometric.nn import aggr
from torch_geometric.transforms import ToUndirected



import time

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'gnn_patch16_224': _cfg(
        crop_pct=0.9, input_size=(3, 224, 224),
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x
    
import torch 
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
        
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))     
        
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p
        
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x
        

class build_unet(nn.Module):
    def __init__(self, out_dim = 1):
        super().__init__()
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)
        self.b = conv_block(512, 1024)         
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)  
        self.outputs = nn.Conv2d(64, out_dim, kernel_size=1, padding=0) 
    def forward(self, inputs):
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)    
        b = self.b(p4)
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        outputs = self.outputs(d4)       
        return outputs

class modStem(nn.Module):
    """ Image to Visual Word Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//8, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//8),
            act_layer(act),
            nn.Conv2d(out_dim//8, out_dim//4, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//4),
            act_layer(act),
            nn.Conv2d(out_dim//4, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x

class gnn_feature_module(nn.Module):
    def __init__(self, out_dim=192) -> None:
        super().__init__()
        self.conv_layer1 = GCNConv(in_channels=3, out_channels=24)
        self.conv_layer2 = GCNConv(in_channels=24, out_channels=48)
        self.conv_layer3 = GCNConv(in_channels=48, out_channels=out_dim)

        self.mean_aggr = aggr.MeanAggregation()

    def forward(self, node_features, edge_indices, batch_indices):
        node_features = self.conv_layer1(node_features, edge_indices)
        node_features = self.conv_layer2(node_features, edge_indices)
        node_features = self.conv_layer3(node_features, edge_indices)
        node_features = self.mean_aggr(node_features, batch_indices)

        return node_features

def vgnn_copy_feature_module(in_dim = 3, out_dim=192):
    aggregator = aggr.MultiAggregation(aggrs=['mean', 'min', 'max', 'std'], mode='proj', mode_kwargs={'in_channels':out_dim, 'out_channels':out_dim})#, 'num_heads':4})
    model = pygnn.Sequential('x, edge_index, batch', [
            (pygnn.conv.GCNConv(in_channels=in_dim, out_channels=out_dim//8), 'x, edge_index -> x'),
            (pygnn.norm.BatchNorm(out_dim//8), 'x -> x'),
            ReLU(inplace=True),
            (pygnn.conv.GCNConv(in_channels=out_dim//8, out_channels=out_dim//4), 'x, edge_index -> x'),
            (pygnn.norm.BatchNorm(out_dim//4), 'x -> x'),
            ReLU(inplace=True),
            (pygnn.conv.GCNConv(in_channels=out_dim//4, out_channels=out_dim//2), 'x, edge_index -> x'),
            (pygnn.norm.BatchNorm(out_dim//2), 'x -> x'),
            ReLU(inplace=True),
            (pygnn.conv.GCNConv(in_channels=out_dim//2, out_channels=out_dim), 'x, edge_index -> x'),
            (pygnn.norm.BatchNorm(out_dim), 'x -> x'),
            ReLU(inplace=True),
            (pygnn.conv.GCNConv(in_channels=out_dim, out_channels=out_dim), 'x, edge_index -> x'),
            (pygnn.norm.BatchNorm(out_dim), 'x -> x'),
            (aggregator, 'x, batch -> x')
            #(pygnn.aggr.MeanAggregation(), 'x, batch -> x')
        ])
    
    return model

def conv_model(out_dim = 24):
    model = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 12, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 12, out_channels = 24, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 24, out_channels = out_dim, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
    return model

def conv_model_large(out_dim = 48):
    model = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 12, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 12, out_channels = 24, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 24, out_channels = 24, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 24, out_channels = out_dim, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 24, out_channels = out_dim, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
    return model


class padding_reshape_layer(nn.Module):
    def __init__(self, dim=192) -> None:
        super().__init__()
        self.dim = dim


    def forward(self, node_features: torch.Tensor, num_sp_list):
        
        # print("GPUS before padding")
        # print(f"currently allocated {torch.cuda.memory_allocated()}")
        # print(f"currently reserved {torch.cuda.memory_reserved()}")
        # print("")
    
        batch_node_features = torch.empty((len(num_sp_list), self.dim, 14, 14)).cuda()
        #print(batch_node_features.is_cuda)
        assert max(num_sp_list) <= 14*14
        l = 0
        for i in range(len(num_sp_list)):
            pad_tensor = torch.zeros(((14*14 - num_sp_list[i]), self.dim)).cuda()
            x = torch.cat((node_features[l:(l+num_sp_list[i]), :], pad_tensor))
            l += num_sp_list[i]
            x = torch.reshape(x, (14, 14, self.dim))
            x = torch.transpose(x, 0, 2)

            #print(f"new shape after transform {x.shape}")

            batch_node_features[i] = x

        #gpus = nvsmi.get_gpus()
        # print("GPUS after padding")
        # print(f"currently allocated {torch.cuda.memory_allocated()}")
        # print(f"currently reserved {torch.cuda.memory_reserved()}")
        # print("")
        #print(list(gpus))
        return batch_node_features


class modDeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(modDeepGCN, self).__init__()
        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        self.n_blocks = opt.n_blocks
        drop_path = opt.drop_path


        self.conv_out_dim = 24
        channels = 64 #192
        self.transform = ToUndirected()




        #self.Stem = modStem(out_dim=channels)

        #self.conv_model = conv_model(out_dim = self.conv_out_dim)
        self.unet = build_unet(self.conv_out_dim)
        self.gnn_feature_module = vgnn_copy_feature_module(in_dim = self.conv_out_dim, out_dim=channels)
        self.padding_reshape_layer = padding_reshape_layer(dim=channels)


        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule 
        print('dpr', dpr)
        num_knn = [int(x.item()) for x in torch.linspace(k, 2*k, self.n_blocks)]  # number of knn's k
        print('num_knn', num_knn)
        max_dilation = 196 // max(num_knn)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, 14, 14))

        if opt.use_dilation:
            self.backbone = Seq(*[Seq(Grapher(channels, num_knn[i], min(i // 4 + 1, max_dilation), conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])
        else:
            self.backbone = Seq(*[Seq(Grapher(channels, num_knn[i], 1, conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])

        self.prediction = Seq(nn.Conv2d(channels, 1024, 1, bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer(act),
                              nn.Dropout(opt.dropout),
                              nn.Conv2d(1024, opt.n_classes, 1, bias=True))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, input):
        #print("FORWARD FUNCTION")
        #print(f"received input {len(input)}")

        forward_start = time.time()
        #gpus = nvsmi.get_gpus()
        # print("GPUS FORWARD begin")
        # #print(list(gpus))
        # print(f"currently allocated {torch.cuda.memory_allocated()}")
        # print(f"currently reserved {torch.cuda.memory_reserved()}")
        # print(f"max allocated {torch.cuda.max_memory_allocated()}")
        # print("")

        #img_b, result_tuples_b, result_indices_b  = input
        img_b, result_tuples_b, catted_index_arrays, num_pixels_list, num_sp_list = input

        #print(f"is cuda img {img_b.is_cuda}")
        # print("GPUS after reading input")
        # #print(list(gpus))
        # print(f"currently allocated {torch.cuda.memory_allocated()}")
        # print(f"currently reserved {torch.cuda.memory_reserved()}")
        # print("")
        # #conv_start = time.time()

        #conv_features = self.conv_model(img_b)
        conv_features = self.unet(img_b)
        #print(f"u net done features are of shape {conv_features.shape}")

        # conv_end = time.time()
        # print(f"conv took {conv_end-conv_start}")

        #print("conv done")
        # print(f"conv first {conv_features[0, :, 0, 0]}")
        # print(f"conv last {conv_features[-1, :, 223, 223]}")

        # print(f"conv features shape is {conv_features.shape}") # (batch_size, out_dim, 224, 224)

        #s1 = time.time()

        # print("GPUS after conv")
        # #print(list(gpus))
        # print(f"currently allocated {torch.cuda.memory_allocated()}")
        # print(f"currently reserved {torch.cuda.memory_reserved()}")
        # print("")

        B, OUT_DIM, H, W = conv_features.shape

        conv_features = torch.transpose(conv_features, 0, 1)
        conv_features = conv_features.contiguous().view(OUT_DIM, B*H*W)

        # print("GPUS after conv transposing")
        # #print(list(gpus))
        # print(f"currently allocated {torch.cuda.memory_allocated()}")
        # print(f"currently reserved {torch.cuda.memory_reserved()}")
        # print("")


        x = torch.index_select(conv_features, 1, catted_index_arrays)
        x = torch.transpose(x, 0, 1)
        splitted = torch.split(x, num_pixels_list, dim=0)





        graph_list = [Data(x=real_features, edge_index=r_t) for r_t, real_features in zip(result_tuples_b, splitted)]
        #print(f"is cuda {graph_list[0].is_cuda}")
        #print(f"is undir {graph_list[0].is_undirected()}")


        #graph_batch_start = time.time()

        graph_batch = Batch.from_data_list(graph_list)

        # print("GPUS after graph batch")
        # #print(list(gpus))
        # print(f"currently allocated {torch.cuda.memory_allocated()}")
        # print(f"currently reserved {torch.cuda.memory_reserved()}")
        # print("")

        node_features = graph_batch["x"]
        edge_indices = graph_batch["edge_index"]
        batch_indices = graph_batch["batch"]

        sp_node_features = self.gnn_feature_module(node_features, edge_indices, batch_indices)

        # print("GPUS after sp node features")
        # #print(list(gpus))
        # print(f"currently allocated {torch.cuda.memory_allocated()}")
        # print(f"currently reserved {torch.cuda.memory_reserved()}")
        # print("")

        padded_and_reshaped_sp_node_features = self.padding_reshape_layer(sp_node_features, num_sp_list)

        #print(f"shape of pd rs tensor is {padded_and_reshaped_sp_node_features.shape}")

        #Vision GNN part
        x = padded_and_reshaped_sp_node_features + self.pos_embed
        B, C, H, W = x.shape

        for i in range(self.n_blocks):
            x = self.backbone[i](x)

        x = F.adaptive_avg_pool2d(x, 1)
        pred = self.prediction(x).squeeze(-1).squeeze(-1)

        #gpus = nvsmi.get_gpus()
        # print("GPUS FORWARD begin")
        # #print(list(gpus))
        # print(f"currently allocated {torch.cuda.memory_allocated()}")
        # print(f"currently reserved {torch.cuda.memory_reserved()}")
        # print("")


        #e4 = time.time()
        #print(f"4 took {e4-s4}")

        forward_end = time.time()
        #print(f"forward took {forward_end-forward_start}")

        return pred





@register_model
def sp_vig(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=100, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.n_blocks = 12 # number of basic blocks in the backbone
            self.n_filters = 192 # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.dropout = drop_rate # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate

    opt = OptInit(**kwargs)
    model = modDeepGCN(opt)
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model
