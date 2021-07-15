import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import time
import cv2
import model.resnet as models
import model.vgg as vgg_models
import util.util as util
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
    return supp_feat


def SA_Weighted_GAP(supp_feat, mask, supp_pred_mask):
    supp_pred = supp_pred_mask+mask
    new_mask1 = torch.zeros_like(mask)
    new_mask2 = torch.zeros_like(mask)

    new_mask1[supp_pred==2] = 1
    new_mask2[supp_pred==1] = 1

    new_mask1[mask==0] = 0
    new_mask2[mask==0] = 0

    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    new_area1 = F.avg_pool2d(new_mask1, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    new_area2 = F.avg_pool2d(new_mask2, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat1 = supp_feat * new_mask1
    supp_feat1 = F.avg_pool2d(input=supp_feat1, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / new_area1
    supp_feat2 = supp_feat * new_mask2
    supp_feat2 = F.avg_pool2d(input=supp_feat2, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / new_area2
    return supp_feat1, supp_feat2
  
def get_vgg16_layer(model):
    layer0_idx = range(0,7)
    layer1_idx = range(7,14)
    layer2_idx = range(14,24)
    layer3_idx = range(24,34)
    layer4_idx = range(34,43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]  
    layer0 = nn.Sequential(*layers_0) 
    layer1 = nn.Sequential(*layers_1) 
    layer2 = nn.Sequential(*layers_2) 
    layer3 = nn.Sequential(*layers_3) 
    layer4 = nn.Sequential(*layers_4)
    return layer0,layer1,layer2,layer3,layer4

class PFENet(nn.Module):
    def __init__(self, layers=50, classes=2, zoom_factor=8, \
        criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, \
        pretrained=True, sync_bn=True, shot=1, ppm_scales=[60, 30, 15, 8], vgg=False):
        super(PFENet, self).__init__()
        assert layers in [50, 101, 152]
        print(ppm_scales)
        assert classes > 1
        from torch.nn import BatchNorm2d as BatchNorm        
        self.zoom_factor = zoom_factor
        self.criterion = criterion
        self.shot = shot
        self.ppm_scales = ppm_scales
        self.vgg = vgg

        models.BatchNorm = BatchNorm
        
        if self.vgg:
            print('INFO: Using VGG_16 bn')
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=pretrained)
            print(vgg16)
            self.layer0, self.layer1, self.layer2, \
                self.layer3, self.layer4 = get_vgg16_layer(vgg16)

        else:
            print('INFO: Using ResNet {}'.format(layers))
            if layers == 50:
                resnet = models.resnet50(pretrained=pretrained)
            elif layers == 101:
                resnet = models.resnet101(pretrained=pretrained)
            else:
                resnet = models.resnet152(pretrained=pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        reduce_dim = 256
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512       

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),                 
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )                 

        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                  
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                   
        )  

        self.pyramid_bins = ppm_scales
        self.avgpool_list = []
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avgpool_list.append(
                    nn.AdaptiveAvgPool2d(bin)
                )


        factor = 1
        mask_add_num = 1
        self.init_merge = []
        self.supp_init_merge = []
        self.beta_conv = []
        self.inner_cls = []        
        for bin in self.pyramid_bins:
            self.supp_init_merge.append(nn.Sequential(
                nn.Conv2d(reduce_dim * 3, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))

            self.init_merge.append(nn.Sequential(
                nn.Conv2d(reduce_dim*3 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))                      
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))            
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),                 
                nn.Conv2d(reduce_dim, classes, kernel_size=1)
            ))            
        self.init_merge = nn.ModuleList(self.init_merge)
        self.supp_init_merge = nn.ModuleList(self.supp_init_merge)
        self.beta_conv = nn.ModuleList(self.beta_conv)
        self.inner_cls = nn.ModuleList(self.inner_cls)                             

        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim*len(self.pyramid_bins), reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),                          
        )              
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),                             
        )                        
     
        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.alpha_conv = []
        for idx in range(len(self.pyramid_bins)-1):
            self.alpha_conv.append(nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU()
            ))     
        self.alpha_conv = nn.ModuleList(self.alpha_conv)

    def forward(self, x, s_x=torch.FloatTensor(1,1,3,473,473).cuda(), s_y=torch.FloatTensor(1,1,473,473).cuda(), y=None):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        #   Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)  
            query_feat_4 = self.layer4(query_feat_3)
            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2),query_feat_3.size(3)), mode='bilinear', align_corners=True)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)

        #   Support Feature     
        supp_feat_list = []
        final_supp_list = []
        mask_list = []
        for i in range(self.shot):
            mask = (s_y[:,i,:,:] == 1).float().unsqueeze(1)
            mask_list.append(mask)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:,i,:,:,:])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
                supp_feat_4 = self.layer4(supp_feat_3*mask)
                final_supp_list.append(supp_feat_4)
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2),supp_feat_3.size(3)), mode='bilinear', align_corners=True)
            
            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat_map = self.down_supp(supp_feat)
            supp_feat = Weighted_GAP(supp_feat_map, mask)
            supp_feat_list.append(supp_feat)

        corr_query_mask_list = []
        cosine_eps = 1e-7
        corr_query_mask = util.compute_prior_mask(final_supp_list, mask_list, query_feat_4, query_feat_3,
                                                  corr_query_mask_list, query_feat, cosine_eps)
        if self.shot > 1:
            supp_feat = supp_feat_list[0]
            for i in range(1, len(supp_feat_list)):
                supp_feat += supp_feat_list[i]
            supp_feat /= len(supp_feat_list)

        supp_out_list = []
        supp_pyramid_feat_list = []

        for idx, tmp_bin in enumerate(self.pyramid_bins):
            if tmp_bin <= 1.0:
                supp_bin = int(supp_feat_map.shape[2] * tmp_bin)
                supp_feat_map_bin = nn.AdaptiveAvgPool2d(supp_bin)(supp_feat_map)
            else:
                bin = tmp_bin
                supp_feat_map_bin = self.avgpool_list[idx](supp_feat_map)

            supp_feat_bin = supp_feat.expand(-1, -1, bin, bin)

            merge_supp_feat_bin = torch.cat([supp_feat_map_bin, supp_feat_bin, supp_feat_bin], 1)
            merge_supp_feat_bin = self.supp_init_merge[idx](merge_supp_feat_bin)

            if idx >= 1:
                pre_supp_feat_bin = supp_pyramid_feat_list[idx-1].clone()
                pre_supp_feat_bin = F.interpolate(pre_supp_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)
                rec_supp_feat_bin = torch.cat([merge_supp_feat_bin, pre_supp_feat_bin], 1)
                merge_supp_feat_bin = self.alpha_conv[idx-1](rec_supp_feat_bin) + merge_supp_feat_bin

            merge_supp_feat_bin = self.beta_conv[idx](merge_supp_feat_bin) + merge_supp_feat_bin
            inner_supp_out_bin = self.inner_cls[idx](merge_supp_feat_bin)
            merge_supp_feat_bin = F.interpolate(merge_supp_feat_bin, size=(supp_feat_map.size(2), supp_feat_map.size(3)), mode='bilinear', align_corners=True)
            supp_pyramid_feat_list.append(merge_supp_feat_bin)
            supp_out_list.append(inner_supp_out_bin)

        supp_feat_map_init = torch.cat(supp_pyramid_feat_list, 1)
        supp_feat_map_init = self.res1(supp_feat_map_init)
        supp_feat_map_init = self.res2(supp_feat_map_init) + supp_feat_map_init
        supp_out = self.cls(supp_feat_map_init)

        supp_pred_mask = torch.argmax(supp_out, dim=1, keepdim=True)
        supp_feat1, supp_feat2 = SA_Weighted_GAP(supp_feat_map, mask, supp_pred_mask)
        if self.training:
            new_supp_out_list = []
            new_supp_pyramid_feat_list = []
            for idx, tmp_bin in enumerate(self.pyramid_bins):
                if tmp_bin <= 1.0:
                    supp_bin = int(supp_feat_map.shape[2] * tmp_bin)
                    supp_feat_map_bin = nn.AdaptiveAvgPool2d(supp_bin)(supp_feat_map)
                else:
                    bin = tmp_bin
                    supp_feat_map_bin = self.avgpool_list[idx](supp_feat_map)

                supp_feat1_bin = supp_feat1.expand(-1, -1, bin, bin)
                supp_feat2_bin = supp_feat2.expand(-1, -1, bin, bin)

                merge_supp_feat_bin = torch.cat([supp_feat_map_bin, supp_feat1_bin, supp_feat2_bin], 1)
                merge_supp_feat_bin = self.supp_init_merge[idx](merge_supp_feat_bin)

                if idx >= 1:
                    # note here we used the previous feature
                    pre_supp_feat_bin = supp_pyramid_feat_list[idx-1].clone()
                    pre_supp_feat_bin = F.interpolate(pre_supp_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)
                    rec_supp_feat_bin = torch.cat([merge_supp_feat_bin, pre_supp_feat_bin], 1)
                    merge_supp_feat_bin = self.alpha_conv[idx-1](rec_supp_feat_bin) + merge_supp_feat_bin

                merge_supp_feat_bin = self.beta_conv[idx](merge_supp_feat_bin) + merge_supp_feat_bin
                inner_supp_out_bin = self.inner_cls[idx](merge_supp_feat_bin)
                merge_supp_feat_bin = F.interpolate(merge_supp_feat_bin, size=(supp_feat_map.size(2), supp_feat_map.size(3)), mode='bilinear', align_corners=True)
                new_supp_pyramid_feat_list.append(merge_supp_feat_bin)
                new_supp_out_list.append(inner_supp_out_bin)

            new_supp_feat_map = torch.cat(new_supp_pyramid_feat_list, 1)
            new_supp_feat_map = self.res1(new_supp_feat_map)
            new_supp_feat_map = self.res2(new_supp_feat_map) + new_supp_feat_map
            new_supp_out = self.cls(new_supp_feat_map)

        out_list = []
        pyramid_feat_list = []

        for idx, tmp_bin in enumerate(self.pyramid_bins):
            if tmp_bin <= 1.0:
                bin = int(query_feat.shape[2] * tmp_bin)
                query_feat_bin = nn.AdaptiveAvgPool2d(bin)(query_feat)
            else:
                bin = tmp_bin
                query_feat_bin = self.avgpool_list[idx](query_feat)

            supp_feat1_bin = supp_feat1.expand(-1, -1, bin, bin)
            supp_feat2_bin = supp_feat2.expand(-1, -1, bin, bin)
            corr_mask_bin = F.interpolate(corr_query_mask, size=(bin, bin), mode='bilinear', align_corners=True)

            merge_feat_bin = torch.cat([query_feat_bin, supp_feat1_bin, supp_feat2_bin, corr_mask_bin], 1)
            merge_feat_bin = self.init_merge[idx](merge_feat_bin)

            if idx >= 1:
                pre_feat_bin = pyramid_feat_list[idx-1].clone()
                pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                merge_feat_bin = self.alpha_conv[idx-1](rec_feat_bin) + merge_feat_bin  

            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin   
            inner_out_bin = self.inner_cls[idx](merge_feat_bin)
            merge_feat_bin = F.interpolate(merge_feat_bin, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)
            pyramid_feat_list.append(merge_feat_bin)
            out_list.append(inner_out_bin)
                 
        query_feat = torch.cat(pyramid_feat_list, 1)
        query_feat = self.res1(query_feat)
        query_feat = self.res2(query_feat) + query_feat           
        out = self.cls(query_feat)

        #   Output Part
        if self.zoom_factor != 1:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
            supp_out = F.interpolate(supp_out, size=(h, w), mode='bilinear', align_corners=True)
            if self.training:
                new_supp_out = F.interpolate(new_supp_out, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            main_loss = self.criterion(out, y.long()) + self.criterion(supp_out, s_y.squeeze(dim=1).long()) \
                        + self.criterion(new_supp_out, s_y.squeeze(dim=1).long())
            aux_loss = torch.zeros_like(main_loss).cuda()
            supp_aux_loss = torch.zeros_like(main_loss).cuda()
            new_supp_aux_loss = torch.zeros_like(main_loss).cuda()
            for idx_k in range(len(out_list)):    
                inner_out = out_list[idx_k]
                inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear', align_corners=True)
                aux_loss = aux_loss + self.criterion(inner_out, y.long())

                supp_inner_out = supp_out_list[idx_k]
                supp_inner_out = F.interpolate(supp_inner_out, size=(h, w), mode='bilinear', align_corners=True)
                supp_aux_loss = supp_aux_loss + self.criterion(supp_inner_out, s_y.squeeze(dim=1).long())

                new_supp_inner_out = new_supp_out_list[idx_k]
                new_supp_inner_out = F.interpolate(new_supp_inner_out, size=(h, w), mode='bilinear', align_corners=True)
                new_supp_aux_loss = new_supp_aux_loss + self.criterion(new_supp_inner_out, s_y.squeeze(dim=1).long())

            aux_loss = aux_loss / len(out_list) + supp_aux_loss / len(supp_out_list) + new_supp_aux_loss / len(new_supp_out_list)
            return out.max(1)[1], main_loss, aux_loss
        else:
            return out





