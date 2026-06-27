from re import sub
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch
from torch import nn
import scipy
from copy import deepcopy
from collections import OrderedDict
from sys import stderr

# for type hint
from torch import Tensor
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# from .swin_transformer import SwinTransformer
from .upernet_pytorch import UPerNet
# from layers.ciconv2d import CIConv2d


import numpy as np
from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
)

def positional_encoding_1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class DPT(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            True,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)


        self.scratch.output_conv = head

    def forward(self, x):
        # if self.channels_last == True:
        #     x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)
        
        return [path_4, path_3, path_2, path_1], out


class DPTDepthModel(DPT):
    def __init__(
        self, path=None, non_negative=True, scale=1.0, shift=0.0, invert=False, **kwargs
    ):
        features = kwargs["features"] if "features" in kwargs else 256

        self.scale = scale
        self.shift = shift
        self.invert = invert

        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)

    def forward(self, x):
        inv_depth = super().forward(x).squeeze(dim=1)

        if self.invert:
            depth = self.scale * inv_depth + self.shift
            depth[depth < 1e-8] = 1e-8
            depth = 1.0 / depth
            return depth
        else:
            return inv_depth


class DPTSegmentationModel(DPT):
    def __init__(self, num_classes, path=None, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256

        kwargs["use_bn"] = True

        head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, **kwargs)

        self.auxlayer = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
        )

        if path is not None:
            self.load(path)

class ProjectionLayer(nn.Sequential):
    def __init__(self, features, n_classes):
        layers = [nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                  nn.BatchNorm2d(features),
                  nn.ReLU(True),
                  nn.Conv2d(features, n_classes, kernel_size=1,bias=True)]
        super().__init__(layers)


class DPTfgsnmodel_hierar(DPT):
    def __init__(self, n_classes, pooling='edge', gfeathead=False, path=None, output_features=True, output_all=False,confidence=False, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256
        self.features = features
        self.output_features = output_features
        self.output_all = output_all
        self.gfeathead = gfeathead
        self.confidence = confidence
        kwargs["use_bn"] = True
        self.centroids = None
        self.prototypical_classifier = False
        self.multiple_maps = False
        self.outputconf = False
        self.confidence_pixloc = False
        head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, n_classes, kernel_size=1),
        )

        super().__init__(head, **kwargs)

        self.auxlayer = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, n_classes, kernel_size=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        if pooling == 'edge':
            self.pooling = EdgePooling()
        elif pooling == 'gpo':
            self.pooling = GPO(16,16)
        elif pooling =='gem':
            self.pooling = GEM()
        elif pooling == 'netvlad':
            self.pooling = NetVLAD()
        elif pooling == 'histogram':
            self.pooling = Histogram()

        if path is not None:
            self.load(path,n_classes)

        projection_layers = []
        for _ in range(3):
            projection_layers.append(nn.Conv2d(features, n_classes, kernel_size=1, padding=0, bias=True))
        # for _ in range(3):
        #     projection_layers.append(ProjectionLayer(features, n_classes))

        if self.confidence:
            self.uncertainty = nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(True),
                nn.Dropout(0.1, False),
                nn.Conv2d(features, 1, kernel_size=1),
            )
            
        self.projection_layers = nn.ModuleList(projection_layers)
        self.interpolate = Interpolate(scale_factor=2, mode="bilinear", align_corners=True)

    def set_uncertainty(self,uncertainty):
        self.uncertainty = uncertainty
        for mlp in self.uncertainty:
            mlp[0].bias = torch.nn.Parameter(torch.ones_like(mlp[0].bias))

        print(self.uncertainty)

    def compute_gfeat(self,h,B):
        N,height, width = h.shape[0], h.shape[-2], h.shape[-1]
        window_size = 50
        overlap_size = 12
        patches = []
        assert(window_size%2==0 & height%2==0 & width%2==0 )

        divsize_x = (width-overlap_size)/(window_size-overlap_size)
        divsize_y = (height-overlap_size)/(window_size-overlap_size)

        (frac,ent) = np.modf(divsize_x)
        step_num_x = int(ent)
        if frac != 0.:
            pad_x = width - step_num_x*(window_size-overlap_size) - overlap_size
        else:
            pad_x = int(frac)
        (frac,ent) = np.modf(divsize_y)
        step_num_y = int(ent)
        if frac != 0.:
            pad_y = width - step_num_y*(window_size-overlap_size) - overlap_size
        else:
            pad_y = int(frac)

        step_x = int(window_size-overlap_size)
        step_y = int(window_size-overlap_size)

        for i in range(step_num_x):
            offset_x = pad_x//2 + i*step_x
            for j in range(step_num_y):
                offset_y = pad_y//2 + j*step_y
                sub_feat = h[:,:,offset_x:offset_x+window_size,offset_y:offset_y+window_size]
                if sub_feat.shape[-2]!=window_size or sub_feat.shape[-2]!=window_size:
                    continue
                patches.append(sub_feat)

        patches = torch.cat(patches,dim=0)
        patches_gdesc = self.pooling(patches) 
        gdesc = torch.stack([torch.reshape(patches_gdesc[i::B],(-1,)) for i in range(B)],dim=0)
        gdesc = torch.nn.functional.normalize(gdesc,p=2,dim=-1)

        return gdesc

    def forward(self, x):
        B = x.shape[0]
        features, h = super().forward(x)
        segmentations = []
        for i,proj in enumerate(self.projection_layers):
            seg = proj(features[i])
            segmentations.append(seg)
        segmentations.append(h)
        if self.confidence:
            uncertainty = self.uncertainty(features[-1])
            conf = torch.sigmoid(-uncertainty)

            if not self.gfeathead:
                gdesc = self.compute_gfeat(conf*h,B)
            else:
                gdesc = self.compute_gfeat(features[-1],B)
        else:
            if not self.gfeathead:
                gdesc = self.compute_gfeat(h,B)
            else:
                gdesc = self.compute_gfeat(features[-1],B)
        if self.training:
            if self.output_all:
                return features, segmentations, gdesc
            elif self.output_features:
                return features, gdesc
            # elif self.train_weights:
            #     conf = []
            #     for i,feat in enumerate(features):
            #         conf.append(self.uncertainty[i](feat))
            #     return segmentations, features, gdesc, conf
            else:
                return segmentations, gdesc

        else:
            if self.output_all:
                return features, segmentations

            elif self.output_features:
                return features[-1], gdesc

            elif self.multiple_maps:
                # return h_feat, aux_feat, gdesc, h_feat
                # return h, self.auxlayer(aux_feat), gdesc, h_feat
                if self.confidence:
                    return segmentations, conf
                else:
                    return segmentations
            # elif self.outputconf:
            #     confidences = []
            #     for uncert_layer, feat_map in zip(self.uncertainty,features):
            #         conf = uncert_layer(feat_map)
            #         # conf = torch.sigmoid(-unc)
            #         confidences.append(conf)
            #     return confidences

            else:
                if self.confidence:
                    return segmentations, features, gdesc, conf
                elif self.confidence_pixloc:
                    confidences = []
                    for uncert_layer, feat_map in zip(self.uncertainty,features[::-1]):
                        unc = uncert_layer(feat_map)
                        conf = torch.sigmoid(-unc)
                        confidences.append(conf)
                    confidences = confidences[::-1] # pixloc order inversed
                    return segmentations, features, gdesc, confidences
                    
                else:
                    return segmentations, features, gdesc


########################################################################################################################################################################################
########################################################################################################################################################################################
########################################################################################################################################################################################
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x




#########






class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=32, dim=100, alpha=10.0,
                 normalize_input=False):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.canny = kornia.filters.Canny(low_threshold=0.04, high_threshold=.05)

        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids[:,:self.dim]).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def get_edge_features(self,feature_map,soft_assign):
        # feature map B,D,H,W
        # output B,max_seq_dim,D

        # ####
        # N,height,width = feature_map.shape[0], feature_map.shape[-2], feature_map.shape[-1]
        # hh = np.arange(1,height+1)
        # ww = np.arange(1,width+1)
        # wg, hg = np.meshgrid(ww, hh)
        # wg = torch.from_numpy(2*(wg/width-0.5)).to('cuda')
        # hg = torch.from_numpy(2*(hg/height-0.5)).to('cuda')
        
        # #####
        # # harm_enc = lambda p : np.concatenate([[np.sin((2**l)*np.pi*p),np.cos((2**l)*np.pi*p)] for l in range(5)])
        # ####
        # feature_map = torch.cat([.1*feature_map,.9*wg.expand(N,1,-1,-1),.9*hg.expand(N,1,-1,-1)],dim=1)

        inp = torch.argmax(feature_map,axis=1,keepdim=True)/255. # B,1,H1,W1
        _, edges = self.canny(inp)
        edges_mask = edges.squeeze(1)>0 # B,H1,W1
        edge_feature = []
        assignements = []

        for b in range(feature_map.shape[0]):
            seq = feature_map[b,:,edges_mask[b]] # d,n_edge_pixels
            assi = soft_assign[b,:,edges_mask[b]] # n_cl,n_edge_pixels
      
            edge_feature.append(seq)
            assignements.append(assi)
         

        return edge_feature,assignements
    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim
        #soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        # x_flatten = x.view(N, C, -1)
        x_flatten = x.reshape(N, C, -1)

       
        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters): # slower than non-looped, but lower memory usage 
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:,C:C+1,:].unsqueeze(2)
            vlad[:,C:C+1,:] = residual.sum(dim=-1)
        
        vlad = F.normalize(vlad, p=2, dim=-1)  # intra-normalization
        vlad = vlad.view(N,-1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=-1)  # L2 normalize
  
        return vlad

    # def forward(self, x):
    #     N, C = x.shape[:2]
    #     desc = []

    #     if self.normalize_input:
    #         x = F.normalize(x, p=2, dim=1)  # across descriptor dim

    #     # soft-assignment
    #     #soft_assign = self.conv(x).view(N, self.num_clusters, -1)
    #     soft_assign = self.conv(x)
    #     edge_feature, soft_assign = self.get_edge_features(x,soft_assign)
        

    #     for b in range(x.shape[0]):
    #         assi = F.softmax(soft_assign[b], dim=1)
    #         x_flatten = edge_feature[b]
    #         print(x_flatten.shape)
    #         # calculate residuals to each clusters
    #         vlad = torch.zeros([self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
    #         for n_cl in range(self.num_clusters): # slower than non-looped, but lower memory usage 
    #             residual = x_flatten.unsqueeze(0) - \
    #                     self.centroids[n_cl:n_cl+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0)
    #             residual *= assi[n_cl:n_cl+1,:].unsqueeze(1)
    #             vlad[n_cl:n_cl+1,:] = residual.sum(dim=-1)
            
    #         vlad = F.normalize(vlad, p=2, dim=-1)  # intra-normalization
    #         vlad = vlad.view(-1)  # flatten
    #         vlad = F.normalize(vlad, p=2, dim=-1)  # L2 normalize
    #         desc.append(vlad)

    #     vlad = torch.vstack(desc)
    #     return vlad


class EdgePooling(nn.Module):
    def __init__(self, window_size=60):
        super(EdgePooling, self).__init__()
        self.canny = kornia.filters.Canny(low_threshold=0.04, high_threshold=0.05)
        self.window_size = window_size

    def forward(self,feature_map):
        height, width = feature_map.shape[-2], feature_map.shape[-1]
        assert(self.window_size%2==0 & height%2==0 & width%2==0 )
        step_num_x, pad_x = height//self.window_size, height%self.window_size
        step_num_y, pad_y = width//self.window_size, width%self.window_size
        gdesc = []
        for i in range(step_num_x):
            offset_x = pad_x//2 + i*self.window_size
            for j in range(step_num_y):
                offset_y = pad_y//2 + j*self.window_size
                sub_feat = feature_map[:,:,offset_x:offset_x+self.window_size,offset_y:offset_y+self.window_size] # B,Nclass,H1,W1
                inp = torch.argmax(sub_feat,axis=1,keepdim=True)/255. # B,1,H1,W1
                _, edges = self.canny(inp)  
                edges_mask = edges.squeeze(1)>0 # B,H1,W1
                window_sumpooled = torch.vstack([torch.sum(sub_feat[b,:,edges_mask[b]],axis=-1) for b in range(inp.shape[0])]) # B,Nclass
                gdesc.append(window_sumpooled)  
                
        gdesc = torch.cat(gdesc,axis=-1)
        gdesc = torch.nn.functional.normalize(gdesc,p=2,dim=-1).view(-1,gdesc.shape[-1])  # B,Nclass*Nwindows

        return gdesc

class GEM(nn.Module):
    def __init__(self,p=3, eps=1e-6):
        super(GEM, self).__init__()
        self.p = nn.Parameter(torch.Tensor([p]))
        self.eps = eps

    def forward(self,feature_map):
      
        gdesc = F.avg_pool2d(feature_map.clamp(min=self.eps).pow(self.p), (feature_map.size(-2), feature_map.size(-1))).pow(1./self.p)
        gdesc = gdesc.squeeze()
        gdesc = torch.nn.functional.normalize(gdesc,p=2,dim=-1).view(-1,gdesc.shape[-1])
        return gdesc

class Histogram(nn.Module):
    def __init__(self):
        super(Histogram, self).__init__()
     

    def forward(self,distrib_map):
        distrib_map = torch.reshape(distrib_map,(distrib_map.shape[0], distrib_map.shape[1],-1)).permute(0,2,1)
        label_map = torch.argmax(distrib_map,dim=-1).to(torch.int64)
        gdesc = []
        for b in range(label_map.shape[0]):
            gdesc.append(torch.bincount(label_map[b],minlength=100)/label_map.shape[-1])
        
        gdesc = torch.stack(gdesc,dim=0).to(torch.float32)
        
        return gdesc

class GPO(nn.Module):
    def __init__(self, d_pe, d_hidden):
        super(GPO, self).__init__()
        self.d_pe = d_pe
        self.d_hidden = d_hidden

        self.pe_database = {}
        self.gru = nn.GRU(self.d_pe, d_hidden, 1, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.d_hidden, 1, bias=False)

        self.canny = kornia.filters.Canny(low_threshold=0.04, high_threshold=.05)

    def get_edge_features(self,feature_map, max_seq_dim=2000):
        # feature map B,D,H,W
        # output B,max_seq_dim,D

        inp = torch.argmax(feature_map,axis=1,keepdim=True)/255. # B,1,H1,W1
        _, edges = self.canny(inp)
        edges_mask = edges.squeeze(1)>0 # B,H1,W1
        lengths = []
        edge_feature = torch.zeros(feature_map.shape[0],feature_map.shape[1],max_seq_dim,device=feature_map.device)
        for b in range(feature_map.shape[0]):
            seq = feature_map[b,:,edges_mask[b]]
            
            if 0<seq.shape[-1]<=max_seq_dim:
                lengths.append(seq.shape[-1])
                edge_feature[b,:,:seq.shape[-1]] = seq
            elif seq.shape[-1]>max_seq_dim:
                lengths.append(max_seq_dim)
                edge_feature[b] = seq[:,:max_seq_dim]
            else:
                lengths.append(1)

        edge_feature = edge_feature.permute(0,2,1)

        return edge_feature,torch.Tensor(lengths).to(feature_map.device)

    def compute_pool_weights(self, lengths, features):
        max_len = int(lengths.max())
        pe_max_len = self.get_pe(max_len)
        pes = pe_max_len.unsqueeze(0).repeat(lengths.size(0), 1, 1).to(lengths.device)
        mask = torch.arange(max_len).expand(lengths.size(0), max_len).to(lengths.device)
        mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1)

        pes = pes.masked_fill(mask == 0, 0)
        self.gru.flatten_parameters()
        packed = pack_padded_sequence(pes, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        padded = pad_packed_sequence(out, batch_first=True)
        out_emb, out_len = padded
        out_emb = (out_emb[:, :, :out_emb.size(2) // 2] + out_emb[:, :, out_emb.size(2) // 2:]) / 2
        scores = self.linear(out_emb)
        scores[torch.where(mask == 0)] = -10000

        weights = torch.softmax(scores / 0.1, 1)
        return weights, mask

    def forward(self, feature_map):
        """
        :param features: features with shape B x K x D
        :param lengths: B x 1, specify the length of each data sample.
        :return: pooled feature with shape B x D
        """
        # features,lengths = self.get_edge_features(feature_map)
        # del feature_map
        #features = feature_map.reshape(feature_map.shape[0],feature_map.shape[1],-1).permute(0,2,1) ## B,D,240,240 -> B,240*240,D
        #features = features[:,:1000]  
        features = feature_map.reshape(feature_map.shape[0],feature_map.shape[1],-1).permute(0,2,1)
        lengths = (torch.ones(features.shape[0])*features.shape[1]).cuda()

        pool_weights, mask = self.compute_pool_weights(lengths, features)
        features = features[:, :int(lengths.max()), :]
        sorted_features = features.masked_fill(mask == 0, -10000)
        sorted_features = sorted_features.sort(dim=1, descending=True)[0]
        sorted_features = sorted_features.masked_fill(mask == 0, 0)

        pooled_features = (sorted_features * pool_weights).sum(1)
        pooled_features = torch.nn.functional.normalize(pooled_features,p=2,dim=-1)

        return pooled_features

    def get_pe(self, length):
        """
        :param length: the length of the sequence
        :return: the positional encoding of the given length
        """
        length = int(length)
        if length in self.pe_database:
            return self.pe_database[length]
        else:
            pe = positional_encoding_1d(self.d_pe, length)
            self.pe_database[length] = pe
            return pe


class ViT(nn.Module):
    def __init__(self, *, num_patches, num_classes, dim, depth, heads, mlp_dim, pool = 'mean', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )

    def forward(self, x):
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        #print(x.shape)
        x = x.view(x.shape[0],-1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        #return self.mlp_head(x)
        return x

def positional_encoding_2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: (height*width)*(d_model*2) position matrix
    """

    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))

    pe_height = torch.zeros(height, d_model)
    pe_width = torch.zeros(width, d_model)
    position_height = torch.arange(0, height).unsqueeze(1)
    position_width = torch.arange(0, width).unsqueeze(1)

    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))
    pe_height[:, 0::2] = torch.sin(position_height.float() * div_term)
    pe_height[:, 1::2] = torch.cos(position_height.float() * div_term)

    pe_width[:, 0::2] = torch.sin(position_width.float() * div_term)
    pe_width[:, 1::2] = torch.cos(position_width.float() * div_term)

    pe_height = pe_height.repeat_interleave(width,dim=0)
    pe_width = pe_width.repeat(height,1)
    pe = torch.cat([pe_height,pe_width],axis=1)
    return pe

class GPO_densesem(nn.Module):
    def __init__(self, d_pe, d_hidden, n_clusters):
        super(GPO_densesem, self).__init__()
        self.d_pe = d_pe
        self.d_hidden = d_hidden

        self.pe_database = {}
        self.gru = nn.GRU(self.d_pe, d_hidden, 1, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.d_hidden, n_clusters, bias=False)


    def compute_pool_weights(self, lengths, height, width):
        max_len = int(lengths.max())
        pe_max_len = self.get_pe(max_len)
        pes = pe_max_len.unsqueeze(0).repeat(lengths.size(0), 1, 1).to(lengths.device)
        mask = torch.arange(max_len).expand(lengths.size(0), max_len).to(lengths.device)
        mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1)

        pes = pes.masked_fill(mask == 0, 0)
        self.gru.flatten_parameters()
        packed = pack_padded_sequence(pes, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        padded = pad_packed_sequence(out, batch_first=True)
        out_emb, out_len = padded
        out_emb = (out_emb[:, :, :out_emb.size(2) // 2] + out_emb[:, :, out_emb.size(2) // 2:]) / 2
        scores = self.linear(out_emb)
        scores[torch.where(mask == 0)] = -10000

        weights = torch.softmax(scores / 0.1, 1)
        return weights, mask

    def forward(self, feature_map):
        """
        :param features: features with shape B x K x D
        :param lengths: B x 1, specify the length of each data sample.
        :return: pooled feature with shape B x D
        """
        # features,lengths = self.get_edge_features(feature_map)
        # print(lengths)
        # del feature_map
        height, width = feature_map.shape[-2],feature_map.shape[-1] 
        features = feature_map.reshape(feature_map.shape[0],feature_map.shape[1],-1).permute(0,2,1) ## B,D,240,240 -> B,240*240,D
        #features = features[:,:1000]  
        lengths = (torch.ones(features.shape[0])*features.shape[1]).cuda()
        pool_weights, mask = self.compute_pool_weights(lengths, height, width)
        features = features[:, :int(lengths.max()), :]
        sorted_features = features.masked_fill(mask == 0, -10000)
        sorted_features = sorted_features.sort(dim=1, descending=True)[0]

        sorted_features = sorted_features.masked_fill(mask == 0, 0)
        pooled_features = (sorted_features * pool_weights).sum(1)
        pooled_features = torch.nn.functional.normalize(pooled_features,p=2,dim=-1)


        return pooled_features

    def get_pe(self, length):
        """
        :param length: the length of the sequence
        :return: the positional encoding of the given length
        """
        length = int(length)
        if length in self.pe_database:
            return self.pe_database[length]
        else:
            pe = positional_encoding_1d(self.d_pe, length)
            self.pe_database[length] = pe
            return pe



