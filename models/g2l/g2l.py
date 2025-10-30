import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ruamel.yaml import YAML
import os 
from typing import List
import timm
import math

def upfeat(input, prob, up_h=2, up_w=2):
    b, c, h, w = input.shape

    # feat_c = input.repeat(1,9,1,1)
    feat = F.unfold(input, 3, 1, 1).reshape(b, -1, h, w)
    # feat = feat - feat_c

    feat = F.interpolate(feat, (h*up_h, w*up_w), mode='nearest').reshape(b, -1, 9, h*up_h, w*up_w)
    feat_sum = (feat*prob.unsqueeze(1)).sum(2)

    # feat_sum = feat_sum + F.interpolate(input, (h*up_h, w*up_w), mode='nearest').reshape(b, -1, h*up_h, w*up_w)

    return feat_sum

    
class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.bias.data.zero_()

class channelAtt(SubModule):
    def __init__(self, cv_chan, im_chan, D=0):
        super(channelAtt, self).__init__()

        self.im_att = nn.Sequential(
            BasicConv(im_chan, im_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(im_chan//2, cv_chan, 1))

        self.weight_init()

    def forward(self, cv, im):
        '''
        '''
        channel_att = self.im_att(im).unsqueeze(2)
        cv = torch.sigmoid(channel_att)*cv
        return cv
    
class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)
        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = self.LeakyReLU(x)#, inplace=True)
        return x
    
class imgChannelAtt(SubModule):
    def __init__(self, im_chan, tar_chan):
        super(imgChannelAtt, self).__init__()

        self.im_att = nn.Sequential(
            BasicConv(im_chan, tar_chan, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(tar_chan, tar_chan, 1))

        self.weight_init()

    def forward(self,im,tar):

        channel_att = self.im_att(im)
        tar = torch.sigmoid(channel_att)*tar
        return tar
    
class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, bn=True, relu=True, keep_dispc=False, att=False):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d 
        if deconv and is_3d: 
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        elif deconv:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=2, padding=1)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=1, padding=1)

        if self.concat: 
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels*2, out_channels*mul, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
            
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

        self.att = att
        if self.att:
            self.im_att = imgChannelAtt(out_channels, out_channels*2)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
            if self.att:
                x = self.im_att(rem, x)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x
    

def conv_bn_relu(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
    padding=(1,1)
    dilation=(1,1)
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=bias),
            nn.ReLU(inplace=True),
        )

class ResBlock(nn.Module):
    def __init__(self, c0, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            BasicConv(c0,c0,kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.Conv2d(c0, c0,kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
        )
        self.bn = nn.BatchNorm2d(c0)
        self.relu = nn.LeakyReLU()

    def forward(self, input):
        x = self.conv(input)
        x = x + input
        x = self.bn(x)
        x = self.relu(x)
        return x


def warp_and_aggregate(disp, bias_list, left, right, group=8, ns=None):
    d_range = torch.arange(right.size(3), device=right.device)
    d_range = d_range.view(1, 1, 1, -1) - disp
    d_range = d_range.repeat(1, right.size(1), 1, 1)
   
    b,c,h,w = left.shape
    left = left.reshape(b, group, c//group, h, w)
    left_norm = left/(torch.norm(left, p=2, dim=2, keepdim=True)+1e-05)

    cost = [ns]
    for offset in bias_list:
        index_float = d_range + offset
        index_long = torch.floor(index_float).long()
        index_left = torch.clip(index_long, min=0, max=right.size(3) - 1)
        index_right = torch.clip(index_long + 1, min=0, max=right.size(3) - 1)
        index_weight = index_float - index_left

        right_warp_left = torch.gather(right, dim=-1, index=index_left.long())
        right_warp_right = torch.gather(right, dim=-1, index=index_right.long())
        right_warp = right_warp_left + index_weight * (right_warp_right - right_warp_left)

        right_warp = right_warp.reshape(b, group, c//group, h, w)
        right_warp_norm = right_warp/(torch.norm(right_warp, p=2, dim=2, keepdim=True)+1e-05)
        cost.append(torch.norm(left_norm-right_warp_norm, p=1, dim=2, keepdim=False))
    cost = torch.cat(cost, dim=1)

    return cost


class LocalDisparityRefinement(SubModule):
    def __init__(self, dilations=[1,2,1,1], bias_list=[2,1,0,-1,-2], group=8):
        super().__init__()
        self.bias_list = bias_list
        self.group = group
        self.in_channels = len(bias_list)*group+group
        self.hidden_channels = self.in_channels
        self.out_channels = len(bias_list)

        self.conv_neighbors = nn.Sequential(BasicConv(in_channels=self.in_channels,out_channels=self.hidden_channels,kernel_size=3,stride=1,padding=1),)
        
        self.res_block = []
        for d in dilations:
            self.res_block += [ResBlock(self.hidden_channels, 3,1,d,d)]
        self.res_block = nn.Sequential(*self.res_block)

        self.conv1 = BasicConv(in_channels=self.hidden_channels,out_channels=self.hidden_channels//4,kernel_size=3,stride=1,padding=1)
        self.convn = nn.Conv2d(self.hidden_channels//4, self.out_channels, 3, 1, 1)

        self.neighborSimilarity = nn.Sequential(BasicConv(in_channels=1, out_channels=group*2, kernel_size=3,stride=1,padding=1),
                                BasicConv(in_channels=group*2, out_channels=group, kernel_size=1,stride=1,padding=0))

        self.weight_init()

    def forward(self, disp, left, right):

        d_expand = disp

        neighborSimilarity = self.neighborSimilarity(d_expand.detach())

        x = warp_and_aggregate(d_expand, self.bias_list, left, right, self.group, neighborSimilarity) #b,12,h/2,w/2
        x = self.conv_neighbors(x)
        x = self.res_block(x)
        x = self.conv1(x)
        x = self.convn(x)

        disp_err_pred = F.softmax(x, dim=1)
        disp_err_ind = torch.tensor(self.bias_list, dtype=torch.float, device=x.device).reshape(1,-1,1,1)
        disp_err = torch.sum(disp_err_pred*disp_err_ind, dim=1, keepdim=True)
        
        d_expand = (d_expand+disp_err)/2
        return d_expand

def regression(cost=None, top_k=None, d=None):
    if cost==None or top_k==None or d==None:
        return None
    
    if top_k == d:
        disp = torch.arange(d).view(1,1,-1,1,1).repeat(cost.shape[0],1,1,cost.shape[-2],cost.shape[-1])
    else:
        _, ind = cost.sort(2, True)
        disp = ind[:, :, :top_k]
        cost = torch.gather(cost, 2, disp)
    
    cost = F.softmax(cost, 2)
    dispmap = torch.sum(cost*disp, 2, keepdim=False)

    return dispmap
    
class Aggregation(SubModule):
    def __init__(self,
                 backbone_cfg,
                 max_disparity=192,
                 matching_head=8,
                 gce=True,
                 disp_strides=2,
                 channels=[16, 32, 48],
                 blocks_num=[2, 2, 2],
                 spixel_branch_channels=[32, 48]
                 ):
        super(Aggregation, self).__init__()
        backbone_type = backbone_cfg['type']
        im_chans = backbone_cfg['channels'][backbone_type]
        self.D = int(max_disparity//4)

        self.conv_stem = BasicConv(matching_head, 8,is_3d=True, kernel_size=3, stride=1, padding=1)

        self.gce = gce

        if gce:
            self.channelAttStem = channelAtt(cv_chan=8,im_chan=80,D=self.D)
            self.channelAtt = nn.ModuleList()
            self.channelAttDown = nn.ModuleList()

        self.conv_down = nn.ModuleList()
        self.conv_up = nn.ModuleList()
        self.conv_skip = nn.ModuleList()
        self.conv_agg = nn.ModuleList()

        channels = [8] + (channels)

        s_disp = disp_strides
        block_n = blocks_num
        inp = channels[0]
        for i in range(3):
            conv: List[nn.Module] = []
            for n in range(block_n[i]):
                stride = (s_disp, 2, 2) if n == 0 else 1
                dilation, kernel_size, padding, bn, relu = 1, 3, 1, True, True
                conv.append(BasicConv(
                    inp, channels[i+1], is_3d=True, bn=bn,
                    relu=relu, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation))
                inp = channels[i+1]
            self.conv_down.append(nn.Sequential(*conv))

            if gce:
                cdfeat_mul = 1 if i == 2 else 2
                self.channelAttDown.append(channelAtt(channels[i+1],cdfeat_mul*im_chans[i+2],self.D//(2**(i+1)),))

            if i == 0:
                out_chan, bn, relu = 1, False, False
            else:
                out_chan, bn, relu = channels[i], True, True
            self.conv_up.append(BasicConv(channels[i+1], out_chan, deconv=True, is_3d=True, bn=bn,
                                          relu=relu, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(s_disp, 2, 2)))

            self.conv_agg.append(nn.Sequential(
                BasicConv(channels[i], channels[i], is_3d=True, kernel_size=3, padding=1, stride=1),
                BasicConv(channels[i], channels[i], is_3d=True, kernel_size=3, padding=1, stride=1),))

            self.conv_skip.append(BasicConv(2*channels[i], channels[i], is_3d=True, kernel_size=1, padding=0, stride=1))

            if gce:
                self.channelAtt.append(channelAtt(channels[i],2*im_chans[i+1],self.D//(2**(i)),))

        self.im_att=nn.ModuleList()
        self.im_att.append(nn.Sequential(
            BasicConv(192, 96, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(96, 64, 1)))
        
        self.im_att.append(nn.Sequential(
            BasicConv(64, 32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(32, 32, 1)))

        self.weight_init()

    def forward(self, img, cost):
        b, c, h, w = img[0].shape

        #对cost提一下特征数量
        cost = cost.reshape(b, -1, self.D, h, w)
        cost = self.conv_stem(cost)
        if self.gce:
            cost = self.channelAttStem(cost, img[0])
        
        #对cost降采样
        cost_feat = [cost]
        cost_ = cost
        for i in range(3):
            cost_ = self.conv_down[i](cost_)
            if self.gce:
                cost_ = self.channelAttDown[i](cost_, img[i+1])
            cost_feat.append(cost_)

        cost_volume=[]
        cost_ = cost_feat[-1]
        for i in range(3):
            cost_ = self.conv_up[-i-1](cost_)
            if cost_.shape != cost_feat[-i-2].shape:
                target_d, target_h, target_w = cost_feat[-i-2].shape[-3:]
                cost_ = F.interpolate(cost_,size=(target_d, target_h, target_w),mode='nearest')
            
            if i == 2:
                cost_volume.append(cost_)
                break
            else:
                #cat att
                img_att=self.im_att[i](img[2-i]).unsqueeze(2)
                img_att=torch.sigmoid(img_att)

                cost_ = torch.cat([cost_, cost_feat[-i-2]], 1)
                cost_ = img_att*cost_
                cost_ = self.conv_skip[-i-1](cost_)
                cost_ = self.conv_agg[-i-1](cost_)

                if self.gce:
                    cost_ = self.channelAtt[-i-1](cost_, img[-i-2])
                cost_volume.append(cost_)

        return cost_volume
     

class CostVolume(nn.Module):
    def __init__(self, maxdisp, glue=False, group=1):
        super(CostVolume, self).__init__()
        self.maxdisp = maxdisp+1
        self.glue = glue
        self.group = group
        self.unfold = nn.Unfold((1, maxdisp+1), 1, 0, 1)
        self.left_pad = nn.ZeroPad2d((maxdisp, 0, 0, 0))

    def forward(self, x, y, v=None):
        b, c, h, w = x.shape

        x_t=x.reshape(b,self.group,-1,1,h,w)
        y_t=self.unfold(self.left_pad(y)).reshape(b,self.group,-1,self.maxdisp,h,w)

        x_norm = x_t/(torch.norm(x_t,p=2,dim=2,keepdim=True)+1e-05)
        y_norm = y_t/(torch.norm(y_t,p=2,dim=2,keepdim=True)+1e-05)

        cost=torch.norm(x_norm-y_norm,p=1,dim=2)
        cost=torch.flip(cost,[2])

        return cost


class AttentionCostVolume(nn.Module):
    def __init__(self, max_disparity, head=1):
        super(AttentionCostVolume, self).__init__()
        
        self.head=head
        self.costVolume = CostVolume(int(max_disparity//4), False, head)   

    def forward(self, imL, imR):

        cost = self.costVolume(imL,imR)

        return cost

    
class FeatUp(SubModule):
    def __init__(self, cfg):
        super(FeatUp, self).__init__()
        self.cfg = cfg['backbone']
        self.type = self.cfg['type']
        chans = self.cfg['channels'][self.type]

        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)
        self.deconv16_8 = Conv2x(chans[3]*2, chans[2], deconv=True, concat=True)
        self.deconv8_4 = Conv2x(chans[2]*2, chans[1], deconv=True, concat=True)
        self.deconv4_2 = Conv2x(chans[1]*2, chans[0], deconv=True, concat=True)

        self.conv4 = BasicConv(chans[1]*2, chans[1]*2, kernel_size=3, stride=1, padding=1)
        self.conv2 = BasicConv(chans[0]*2, chans[0]*2, kernel_size=3, stride=1, padding=1)

        self.deconv2_1 = Conv2x(32, 16, deconv=True)

        self.weight_init()

    def forward(self, featL, featR=None, freezen_disp4=False):
        x, x2, x4, x8, x16, x32 = featL
        if freezen_disp4:
            with torch.no_grad():
                x16 = self.deconv32_16(x32, x16)
                x8 = self.deconv16_8(x16, x8)
                x4 = self.deconv8_4(x8, x4)
            x2 = self.deconv4_2(x4 ,x2)
        else:
            x16 = self.deconv32_16(x32, x16)
            x8 = self.deconv16_8(x16, x8)
            x4 = self.deconv8_4(x8, x4)
            x2 = self.deconv4_2(x4 ,x2)
        return [x, x2, x4, x8, x16, x32]


class Feature(SubModule):
    def __init__(self, cfg):
        super(Feature, self).__init__()
        self.cfg = cfg['backbone']
        self.type = self.cfg['type']
        chans = self.cfg['channels'][self.type]
        layers = self.cfg['layers'][self.type]

        pretrained = False if self.cfg['from_scratch'] else True
        model = timm.create_model(self.type, pretrained=pretrained, features_only=True)

        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1

        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])

        self.stem = nn.Sequential(
            BasicConv(3, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU()
            )
        
        self.stem_1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x, freezen_disp4=False):
        if freezen_disp4:
            with torch.no_grad():
                x = self.bn1(self.conv_stem(x))
                x2 = self.block0(x)
                x4 = self.block1(x2)

                x8 = self.block2(x4)
                x16 = self.block3(x8)
                x32 = self.block4(x16)
        else:
            x = self.bn1(self.conv_stem(x))
            x2 = self.block0(x)
            x4 = self.block1(x2)

            x8 = self.block2(x4)
            x16 = self.block3(x8)
            x32 = self.block4(x16)

        x_out = [x, x2, x4, x8, x16, x32]

        return x_out

class ConfidenceEstimation(SubModule):

    def __init__(self, in_planes, batchNorm=True):
        super(ConfidenceEstimation, self).__init__()

        self.in_planes = in_planes
        self.sec_in_planes = int(self.in_planes//3)
        self.sec_in_planes  = self.sec_in_planes if self.sec_in_planes > 0 else 1

        self.conf_net = nn.Sequential(conv_bn_relu(batchNorm, self.in_planes, self.sec_in_planes, 3, 1, 1, bias=False),
                                      nn.Conv2d(self.sec_in_planes, 1, 1, 1, 0, bias=False))
        self.weight_init()

    def forward(self, cost):
        assert cost.shape[1] == self.in_planes
        confCost = self.conf_net(cost)
        confCost = torch.sigmoid(confCost)

        return confCost

def get_disp_conf(input,up_h=2, up_w=2):
    #b,1,h,w
    b,_,h,w = input.shape
    input = F.unfold(input,3,1,1).reshape(b,-1,h,w)
    input = F.interpolate(input, size=(h*up_h,w*up_w), mode='nearest').reshape(b,9,h*up_h,w*up_w)
    return input

current_file_directory = os.path.dirname(__file__)
current_working_directory = os.getcwd()
relative_path = os.path.relpath(current_file_directory, current_working_directory)

def load_configs(path):
    cfg = YAML().load(open(path, 'r'))
    backbone_cfg = YAML().load(open(os.path.join(relative_path,cfg['backbone']['cfg_path']), 'r'))
    cfg['backbone'].update(backbone_cfg)
    return cfg


class G2l(nn.Module):
    def __init__(self, only_disp4=False, freezen_disp4=False):
        super(G2l, self).__init__()
        cfg=load_configs('{}/configs/stereo/cfg_coex.yaml'.format(relative_path))
        self.cfg = cfg

        self.D = int(self.cfg['max_disparity']/4)
        self.top_k = 2
        self.only_disp4 = only_disp4
        self.freezen_disp4 = freezen_disp4

        #主干特征
        self.feature = Feature(self.cfg)
        self.up = FeatUp(self.cfg)

        #成本体
        self.cost_volume = AttentionCostVolume(max_disparity=192,head=8)
        self.cost_agg = Aggregation(
            cfg['backbone'],
            max_disparity=cfg['max_disparity'],
            matching_head=8,
            gce=cfg['gce'],
            disp_strides=cfg['aggregation']['disp_strides'],
            channels=cfg['aggregation']['channels'],
            blocks_num=cfg['aggregation']['blocks_num'],
            spixel_branch_channels=cfg['spixel']['branch_channels'])
        
        self.preprocess_2 = nn.Sequential(BasicConv(in_channels=80, out_channels=48, kernel_size=3, padding=1, stride=1),
                                        nn.Conv2d(in_channels=48, out_channels=48, kernel_size=1, padding=0, stride=1))
        
        self.preprocess_1 = nn.Sequential(BasicConv(in_channels=48, out_channels=32, kernel_size=3, padding=1, stride=1),
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0, stride=1))
        #分支特征
        self.stem_0 = nn.Sequential(
            BasicConv(3, 8, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(8, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU()
            )        
        self.stem_1 = nn.Sequential(
            BasicConv(8, 16, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU()
            )
        self.stem_2 = nn.Sequential(
            BasicConv(16, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
            )
        
        #超像素预测
        #4预测2
        self.spx0_2 = nn.Sequential(
            BasicConv(80, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.BatchNorm2d(24), nn.ReLU()
            ) 
        self.spx0_1 = Conv2x(in_channels=24, out_channels=16, deconv=True)
        self.spx0 = nn.Sequential(nn.Conv2d(2*16, 9, kernel_size=3, stride=1, padding=1))

        #2预测1
        self.spx_2 = nn.Sequential(
            BasicConv(48, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU()
            )   
        self.spx_1 = Conv2x(in_channels=16, out_channels=8, deconv=True)
        # self.spx = nn.Sequential(nn.ConvTranspose2d(2*24, 9, kernel_size=4, stride=2, padding=1))
        self.spx = nn.Sequential(nn.Conv2d(2*8, 9, kernel_size=3, stride=1, padding=1))

        #细化视差图
        self.local_disparity_refinement = LocalDisparityRefinement()

        #置信度
        self.conf_est = ConfidenceEstimation(self.D)
        
    def forward(self, imL, imR=None, u0=None, v0=None, training=False):
        if imR is not None:
            assert imL.shape == imR.shape
            imL = torch.cat([imL, imR], 0) #concat到一起提取特征，后续分开
        b,c,h,w = imL.shape #8,3,256,512

        #主干特征
        v = self.feature(imL,freezen_disp4=self.freezen_disp4) #v2 8,16,128,256 + v 8,24,64,128\8,32,32,64\8,96,16,32\8,160,8,16
        v = self.up(v,freezen_disp4=self.freezen_disp4) #v 8,48,64,128\8,64,32,64\8,192,16,32\8,160,8,16
        x, y = [], []
        for v_ in v:
            x_, y_ = v_.split(dim=0, split_size=b//2)
            x.append(x_)
            y.append(y_)

        if self.freezen_disp4:
            with torch.no_grad():
                #超像素分支特征
                stem_0v = self.stem_0(imL)
                stem_0x, stem_0y = stem_0v.split(dim=0, split_size=b//2)  
                stem_1v = self.stem_1(stem_0v)
                stem_1x, stem_1y = stem_1v.split(dim=0, split_size=b//2)
                stem_2v = self.stem_2(stem_1v)
                stem_2x, stem_2y = stem_2v.split(dim=0, split_size=b//2)

                #合并特征
                x[2] = torch.cat((x[2], stem_2x), 1) #4,96,64,128
                y[2] = torch.cat((y[2], stem_2y), 1) #4,96,64,128
                left = self.preprocess_2(x[2])
                right = self.preprocess_2(y[2])

                #构建成本体
                cost = (self.cost_volume(left, right))[:, :, :-1] #4,1,48,64,128,包含0视差
                #成本聚合
                costs = self.cost_agg(x[2:], cost) #4,1,48,64,128

                #置信度预测
                disp_conf_o = self.conf_est(costs[-1].squeeze(1))
                disp_conf = get_disp_conf(disp_conf_o)

                xspx0 = self.spx0_2(x[2])
                xspx0 = self.spx0_1(xspx0, stem_1x)
                spx0_pred = self.spx0(xspx0)
                spx0_pred = spx0_pred*disp_conf
                spx0_pred = F.softmax(spx0_pred, 1)

                #计算视差图
                disp_4 = regression(costs[-1], self.top_k, self.D)
                disp_4_expand = upfeat(disp_4, spx0_pred, 2, 2)*2
                
                costs.append(disp_conf_o)


            x[1] = torch.cat((x[1], stem_1x), 1) #4,96,64,128
            y[1] = torch.cat((y[1], stem_1y), 1) #4,96,64,128
            left_fine = self.preprocess_1(x[1])
            right_fine = self.preprocess_1(y[1]) 
            disp_2 = self.local_disparity_refinement(disp_4_expand, left_fine, right_fine)

            xspx = self.spx_2(x[1])
            xspx = self.spx_1(xspx, stem_0x)
            spx_pred = self.spx(xspx) #4,9,256,512
            spx_pred = F.softmax(spx_pred, 1) #4,9,256,512

            disp_1 = upfeat(disp_2, spx_pred, 2, 2)
            disp_1 = disp_1.squeeze(1)*4
            disp_2 = disp_2.squeeze(1)*4
            disp_4_expand = disp_4_expand.squeeze(1)*2
            disp_4 = disp_4.squeeze(1)*4
        
        else:
            #超像素分支特征
            stem_0v = self.stem_0(imL)
            stem_0x, stem_0y = stem_0v.split(dim=0, split_size=b//2)  
            stem_1v = self.stem_1(stem_0v)
            stem_1x, stem_1y = stem_1v.split(dim=0, split_size=b//2)
            stem_2v = self.stem_2(stem_1v)
            stem_2x, stem_2y = stem_2v.split(dim=0, split_size=b//2)

            #合并特征
            x[2] = torch.cat((x[2], stem_2x), 1) #4,96,64,128
            y[2] = torch.cat((y[2], stem_2y), 1) #4,96,64,128
            left = self.preprocess_2(x[2])
            right = self.preprocess_2(y[2])

            #构建成本体
            cost = (self.cost_volume(left, right))[:, :, :-1] #4,1,48,64,128,包含0视差
            #成本聚合
            costs = self.cost_agg(x[2:], cost) #4,1,48,64,128

            #置信度预测
            disp_conf_o = self.conf_est(costs[-1].squeeze(1))
            disp_conf = get_disp_conf(disp_conf_o)

            xspx0 = self.spx0_2(x[2])
            xspx0 = self.spx0_1(xspx0, stem_1x)
            spx0_pred = self.spx0(xspx0)
            spx0_pred = spx0_pred*disp_conf
            spx0_pred = F.softmax(spx0_pred, 1)

            #计算视差图
            disp_4 = regression(costs[-1], self.top_k, self.D)
            disp_4_expand = upfeat(disp_4, spx0_pred, 2, 2)*2
            
            costs.append(disp_conf_o)

            if not self.only_disp4:
                x[1] = torch.cat((x[1], stem_1x), 1) #4,96,64,128
                y[1] = torch.cat((y[1], stem_1y), 1) #4,96,64,128
                left_fine = self.preprocess_1(x[1])
                right_fine = self.preprocess_1(y[1]) 
                disp_2 = self.local_disparity_refinement(disp_4_expand, left_fine, right_fine)

                xspx = self.spx_2(x[1])
                xspx = self.spx_1(xspx, stem_0x)
                spx_pred = self.spx(xspx) #4,9,256,512
                spx_pred = F.softmax(spx_pred, 1) #4,9,256,512

                disp_1 = upfeat(disp_2, spx_pred, 2, 2)
                disp_1 = disp_1.squeeze(1)*4
                disp_2 = disp_2.squeeze(1)*4
                disp_4_expand = disp_4_expand.squeeze(1)*2
                disp_4 = disp_4.squeeze(1)*4
            else:
                disp_1 = F.interpolate(disp_4_expand, size=(h,w), mode='bilinear').squeeze(1)*2
                disp_4 = disp_4.squeeze(1)*4
                disp_4_expand = disp_4_expand.squeeze(1)*2
                disp_2 = disp_4_expand
                

        return [disp_1,disp_2,disp_4_expand,disp_4,costs]

    



