# SCL-PFENet
We follow the same setting with the paper [**PFENet: Prior Guided Feature Enrichment Network for Few-shot Segmentation**](http://arxiv.org/abs/2008.01449). 

This file is mainly follow the same instruction with [PFENet](https://github.com/dvlab-research/PFENet) 

All our final model can be found in [here](https://drive.google.com/drive/folders/1tkLCJ9j8rsJDgehcHNR-zWd0yaEiDPHR?usp=sharing)

We used one 2080Ti with pytorch 1.3/1.4 (1.3 seems more stable).
# Get Started

### Environment
+ torch==1.4.0 (torch version >= 1.0.1.post2 should be okay to run this repo)
+ numpy==1.18.4
+ tensorboardX==1.8
+ cv2==4.2.0


### Datasets and Data Preparation

Please download the following datasets:

+ PASCAL-5i is based on the [**PASCAL VOC 2012**](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) and [**SBD**](http://home.bharathh.info/pubs/codes/SBD/download.html) where the val images should be excluded from the list of training samples.

+ [**COCO 2014**](https://cocodataset.org/#download).

This code reads data from .txt files where each line contains the paths for image and the correcponding label respectively. Image and label paths are seperated by a space. Example is as follows:

    image_path_1 label_path_1
    image_path_2 label_path_2
    image_path_3 label_path_3
    ...
    image_path_n label_path_n

Then update the train/val/test list paths in the config files.

#### [Update] We have uploaded the lists we use in our paper.
+ The train/val lists for COCO contain 82081 and 40137 images respectively. They are the default train/val splits of COCO. 
+ The train/val lists for PASCAL5i contain 5953 and 1449 images respectively. The train list should be **voc_sbd_merge_noduplicate.txt** and the val list is the original val list of pascal voc (**val.txt**).

##### To get voc_sbd_merge_noduplicate.txt:
+ We first merge the original VOC (voc_original_train.txt) and SBD ([**sbd_data.txt**](http://home.bharathh.info/pubs/codes/SBD/train_noval.txt)) training data. 
+ [**Important**] sbd_data.txt does not overlap with the PASCALVOC 2012 validation data.
+ The merged list (voc_sbd_merge.txt) is then processed by the script (duplicate_removal.py) to remove the duplicate images and labels.

### Run Demo / Test with Pretrained Models
+ Please download the pretrained models.
+ Update the config file by speficifying the target **split** and **path** (`weights`) for loading the checkpoint.
+ Execute `mkdir initmodel` at the root directory.
+ Download the ImageNet pretrained [**backbones**](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155122171_link_cuhk_edu_hk/EQEY0JxITwVHisdVzusEqNUBNsf1CT8MsALdahUhaHrhlw?e=4%3a2o3XTL&at=9) and put them into the `initmodel` directory.
+ Then execute the command: 

    `sh test.sh {*dataset*} {*model_config*}`

Example: Test PFENet with ResNet50 on the split 0 of PASCAL-5i: 

    sh test.sh pascal split0_resnet50


### Train

Execute this command at the root directory: 

    sh train.sh {*dataset*} {*model_config*}

### Note

In "PFENet_SCL_1shot.py", Line 310, we used the previous feature map instead of the current feature map.
