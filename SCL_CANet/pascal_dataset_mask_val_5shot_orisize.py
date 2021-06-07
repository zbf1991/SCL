import random
import os
import torchvision
import torch
from PIL import Image
import torchvision.transforms.functional as F
import torch.nn.functional as F_tensor
import numpy as np
from torch.utils.data import DataLoader
import time


class Dataset(object):


    def  __init__(self, data_dir, fold, input_size=[321, 321], normalize_mean=[0, 0, 0],
                 normalize_std=[1, 1, 1]):

        self.data_dir = data_dir

        self.input_size = input_size

        #random sample 1000 pairs
        self.chosen_data_list_1 = self.get_new_exist_class_dict(fold=fold)
        chosen_data_list_2 = self.chosen_data_list_1[:]
        chosen_data_list_3 = self.chosen_data_list_1[:]
        random.shuffle(chosen_data_list_2)
        random.shuffle(chosen_data_list_3)
        self.chosen_data_list=self.chosen_data_list_1+chosen_data_list_2+chosen_data_list_3
        self.chosen_data_list=self.chosen_data_list[:1000]

        self.binary_pair_list = self.get_binary_pair_list()#a dict of each class, which contains all imgs that include this class
        self.history_mask_list = [None] * 1000
        self.query_class_support_list=[None] * 1000
        for index in range(1000):
            query_name=self.chosen_data_list[index][0]
            sample_class=self.chosen_data_list[index][1]
            support_img_list = self.binary_pair_list[sample_class]  # all img that contain the sample_class

            selected_support_img = []
            while True:  # random sample a support data
                support_name = support_img_list[random.randint(0, len(support_img_list) - 1)]
                if support_name != query_name:
                    support_flag = True
                    for added_support_name in selected_support_img:
                        if support_name == added_support_name:
                            support_flag = False
                    if support_flag == True:
                        selected_support_img.append(support_name)
                if len(selected_support_img) > 4:
                    break
            self.query_class_support_list[index]=[query_name,sample_class,selected_support_img]



        self.initiaize_transformation(normalize_mean, normalize_std, input_size)
        pass

    def get_new_exist_class_dict(self, fold):
        new_exist_class_list = []

        f = open(os.path.join(self.data_dir, 'Binary_map_aug','val', 'split%1d_val.txt' % (fold)))
        while True:
            item = f.readline()
            if item == '':
                break
            img_name = item[:11]
            cat = int(item[13:15])
            new_exist_class_list.append([img_name, cat])
        return new_exist_class_list


    def initiaize_transformation(self, normalize_mean, normalize_std, input_size):
        self.ToTensor = torchvision.transforms.ToTensor()
        self.normalize = torchvision.transforms.Normalize(normalize_mean, normalize_std)

    def get_binary_pair_list(self):  # a list store all img name that contain that class
        binary_pair_list = {}
        for Class in range(1, 21):
            binary_pair_list[Class] = self.read_txt(
                os.path.join(self.data_dir, 'Binary_map_aug', 'val', '%d.txt' % Class))
        return binary_pair_list

    def read_txt(self, dir):
        f = open(dir)
        out_list = []
        line = f.readline()
        while line:
            out_list.append(line.split()[0])
            line = f.readline()
        return out_list

    def __getitem__(self, index):

        # give an query index,sample a target class first
        query_name = self.query_class_support_list[index][0]
        sample_class = self.query_class_support_list[index][1]  # random sample a class in this img
        support_name_list=self.query_class_support_list[index][2]

        support_rgb_list = []
        support_mask_list = []
        # random scale and crop for support
        for support_name in support_name_list:
            support_rgb = self.normalize(self.ToTensor(
                Image.open(os.path.join(self.data_dir, 'JPEGImages', support_name + '.jpg'))))
            support_mask = self.ToTensor(Image.open(os.path.join(self.data_dir, 'Binary_map_aug', 'val', str(sample_class),
                                                   support_name + '.png')))
            support_rgb_list.append(support_rgb)
            support_mask_list.append(support_mask)


        query_rgb = self.normalize(self.ToTensor(
            Image.open(os.path.join(self.data_dir, 'JPEGImages', query_name + '.jpg'))))
        query_mask = self.ToTensor(Image.open(os.path.join(self.data_dir, 'Binary_map_aug', 'val', str(sample_class),
                                           query_name + '.png')))

        if self.history_mask_list[index] is None:

            history_mask=torch.zeros(2,41,41).fill_(0.0)

        else:

            history_mask=self.history_mask_list[index]


        return query_rgb, query_mask, support_rgb_list, support_mask_list,history_mask,sample_class,index

    def flip(self, flag, img):
        if flag > 0.5:
            return F.hflip(img)
        else:
            return img

    def __len__(self):
        return 1000
