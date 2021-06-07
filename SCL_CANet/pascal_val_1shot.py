from torch.utils import data
import torch.backends.cudnn as cudnn
from utils import *
import torch.nn.functional as F
import argparse
from pascal_dataset_mask_val_orisize import Dataset as Dataset_val
import os
import torch
from SCL_1shot import Res_Deeplab
import torch.nn as nn
import numpy as np

parser = argparse.ArgumentParser()


parser.add_argument('-bs',type=int,help='batchsize',default=1)
parser.add_argument('-bs_val',type=int,help='batchsize for val',default=1)
parser.add_argument('-fold',type=int,help='fold',default=0)
parser.add_argument('-gpu',type=str,help='gpu id to use',default='0')
parser.add_argument('-iter_time',type=int,default=5)

options = parser.parse_args()


data_dir = 'path/to/VOCdevkit/VOC2012/'

#set gpus
# gpu_list = [int(x) for x in options.gpu.split(',')]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu

torch.backends.cudnn.benchmark = True


IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
num_class = 2
num_epoch = 200
input_size = (321, 321)
batch_size = options.bs
weight_decay = 0.0005
momentum = 0.9
power = 0.9

cudnn.enabled = True


# Create network.
model = Res_Deeplab(num_classes=num_class)
model=nn.DataParallel(model)
model_name = './checkpoint/fo=' + str(options.fold) + '/model/best.pth'
model.load_state_dict(torch.load(model_name))


# disable the  gradients of not optomized layers
val_turn_off(model)

checkpoint_dir = 'checkpoint/fo=%d/'% options.fold
check_dir(checkpoint_dir)

valset = Dataset_val(data_dir=data_dir, fold=options.fold, input_size=input_size, normalize_mean=IMG_MEAN,
                 normalize_std=IMG_STD)
valloader = data.DataLoader(valset, batch_size=options.bs_val, shuffle=False, num_workers=4,drop_last=False)


iou_list = []#track validaiton iou
highest_iou = 0

model.cuda()

model = model.eval()

valset.history_mask_list=[None] * 1000
best_iou = 0

for eva_iter in range(options.iter_time):
    all_inter, all_union, all_predict = [0] * 5, [0] * 5, [0] * 5
    num =0

    for i_iter, batch in enumerate(valloader):

        query_rgb, query_mask, support_rgb, support_mask, history_mask, sample_class, index = batch
        query_rgb = (query_rgb).cuda(0)
        support_rgb = (support_rgb).cuda(0)
        support_mask = (support_mask).cuda(0)
        query_mask = (query_mask).cuda(0).long()  # change formation for crossentropy use

        query_mask = query_mask[:, 0, :, :]  # remove the second dim,change formation for crossentropy use
        history_mask = (history_mask).cuda(0)
        pred_softmax = torch.zeros(1, 2, query_rgb.size(-2), query_rgb.size(-1)).cuda(0)

        pred = torch.zeros(1, 2, query_rgb.size(-2), query_rgb.size(-1)).cuda(0)
        for scale in [0.7, 1, 1.3]:
            query_ = nn.functional.interpolate(query_rgb, scale_factor=scale, mode='bilinear', align_corners=False)
            scale_pred = model(query_, support_rgb, support_mask, history_mask)[0]

            pred_softmax += nn.functional.interpolate(scale_pred, size=query_rgb.size()[-2:], mode='bilinear',
                                                      align_corners=False)
        # pred_softmax /= 3.
        pred_softmax = F.softmax(pred_softmax, dim=1)

        history_pred_softmax = pred_softmax.clone().data.cpu()
        # update history mask
        for j in range(support_mask.shape[0]):
            sub_index = index[j]
            valset.history_mask_list[sub_index] = history_pred_softmax[j]

        _, pred_label = torch.max(pred_softmax, 1)

        inter_list, union_list, _, num_predict_list = get_iou_v1(query_mask, pred_label)
        for j in range(query_mask.shape[0]):  # batch size
            all_inter[sample_class[j] - (options.fold * 5 + 1)] += inter_list[j]
            all_union[sample_class[j] - (options.fold * 5 + 1)] += union_list[j]

        if num%500==0:
            print(num)
        num+=1


    IOU = [0] * 5

    for j in range(5):
        IOU[j] = all_inter[j] / all_union[j]
        print(j," IOU is ",IOU[j])

    mean_iou = np.mean(IOU)
    print('IOU:%.4f' % (mean_iou))






