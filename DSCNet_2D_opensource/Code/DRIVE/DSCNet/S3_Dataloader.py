# -*- coding: utf-8 -*-
import os
import torch
from torch.utils import data
import numpy as np
import SimpleITK as sitk
from S3_Data_Augumentation import transform_img_lab
import tifffile
import warnings

warnings.filterwarnings("ignore")


def read_file_from_txt(txt_path):
    files = []
    for line in open(txt_path, "r"):
        filepath = line.strip()
        if not filepath.endswith('.tif') and not filepath.endswith('.tiff'):
            # 跳过非图片文件
            continue
        if not os.path.isfile(filepath):
            # 尝试替换后缀
            filepath = filepath.replace('.tif', '.tiff') if filepath.endswith('.tif') else filepath.replace('.tiff', '.tif')
        if os.path.isfile(filepath):
            files.append(filepath)
        else:
            # 如果文件仍然不存在，打印警告信息
            print(f"Warning: 文件 {filepath} 不存在。")
    return files

class Dataloader(data.Dataset):
    def __init__(self, args):
        super(Dataloader, self).__init__()
        self.image_file = read_file_from_txt(args.Image_Tr_txt)
        self.label_file = read_file_from_txt(args.Label_Tr_txt)
        self.shape = (args.ROI_shape, args.ROI_shape)
        self.args = args

    def __getitem__(self, index):
        
        # 用来读取tiff文件
        image = tifffile.imread(self.image_file[index])
        label = tifffile.imread(self.label_file[index])

        # 确保图像是三维的，如果是二维图像，那么我们需要添加一个额外的维度
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        # 如果图像有多于1个通道，那么将其转换为灰度图
        if image.shape[2] > 1:
            # 这里是将RGB转换为灰度图的标准方法，您可以根据需要调整权重
            image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
            image = image.astype(np.float32)
        
        y, x = image.shape
        
        # Normalization
        mean, std = np.load(self.args.root_dir + self.args.Tr_Meanstd_name)
        image = (image - mean) / std
        label = label / (np.max(label))

        # Random crop, (center_y, center_x) refers the left-up coordinate of the Random_Crop_Block
        center_y = np.random.randint(0, y - self.shape[0] + 1, 1, dtype=np.int16)[0]
        center_x = np.random.randint(0, x - self.shape[1] + 1, 1, dtype=np.int16)[0]
        image = image[
            center_y : self.shape[0] + center_y, center_x : self.shape[1] + center_x
        ]
        label = label[
            center_y : self.shape[0] + center_y, center_x : self.shape[1] + center_x
        ]

        image = image[np.newaxis, :, :]
        label = label[np.newaxis, :, :]

        # Data Augmentation
        data_dict = transform_img_lab(image, label, self.args)
        image_trans = data_dict["image"]
        label_trans = data_dict["label"]
        if isinstance(image_trans, torch.Tensor):
            image_trans = image_trans.numpy()
        if isinstance(label_trans, torch.Tensor):
            label_trans = label_trans.numpy()

        return image_trans, label_trans

    def __len__(self):
        return len(self.image_file)
