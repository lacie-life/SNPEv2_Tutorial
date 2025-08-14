"""
The file defines prediction process to compare with dlc results.
2022.11.24
@Author: Youn Joo Lee
"""

from utils.helpers import get_dataset_info, check_related_path, get_colored_info, color_encode
from utils.losses import categorical_crossentropy_with_logits
from utils.layers import GlobalAveragePooling2D, Concatenate
from utils.utils import load_image, decode_one_hot
from utils.metrics import MeanIoU
from config import cfg
from PIL import Image
from tensorflow.keras.applications import imagenet_utils

import tensorflow as tf
import numpy as np
import sys
import cv2
import os
import natsort

os.environ["CUDA_VISIBLE_DEVICES"]="2"

# get image and label file names for training and validation
_, _,  _,_,test_image_names, test_label_names= get_dataset_info(cfg.MODEL.DATASET)
#test_image_names = os.listdir('./output')
print(len(test_image_names))
# get color info
#csv_file = os.path.join('../Datasets/Fire/flame', 'class_dict.csv')
csv_file = './flame/class_dict.csv'
_, color_values = get_colored_info(csv_file)
print(color_values)
test_image_names = natsort.natsorted(test_image_names)

output_dir = './predictions/V3Plus/output_prune/flame_AND_90'
save_dir = './predictions/V3Plus/output_prune/flame_AND_90%_inference' #'./flood/tong'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for n, name in enumerate(test_image_names):
    file_name = output_dir + '/Result_' + f'{n}' + '/DeepLabV3Plus/up_sampling2d_2/resize/ResizeBilinear:0.raw'  # output된 raw데이터 디렉토리 설정
    #file_name = output_dir + '/Result_0/DeepLabV3/up_sampling2d_1/resize/ResizeBilinear.raw'  # output된 raw데이터 디렉토리 설정

    fid = open(file_name,'r')
    dlc_output = np.fromfile(fid, dtype=np.float32)
    dlc_output = np.reshape(dlc_output, [512, 512, 2])
    
    
    # decode one-hot
    dlc_output = decode_one_hot(dlc_output)

     #영상확인할 때!!! color_encode 주석처리를 해제하면 저장할 때 영상분할 결과를 컬러로 확인할 수 있어요!
    #color encode
    dlc_output = color_encode(dlc_output, color_values)
     # get PIL file
    dlc_output = Image.fromarray(np.uint8(dlc_output))
    

    # save the prediction
    _, test_file_name = os.path.split(name)

    #dlc_output.save('./predictions/flame/qat_exam5/pred_n/' + test_file_name[:-4] + '.png')
    dlc_output.save(os.path.join(save_dir ,test_file_name[:-4]) + '.png')
# file_name = './output/Result_0/DeepLabV3/up_sampling2d_1/resize/ResizeBilinear.raw'
# fid = open(file_name,'r')
# dlc_output = np.fromfile(fid, dtype=np.float32)
# dlc_output = np.reshape(dlc_output, [256, 256, 2])
# dlc_output = decode_one_hot(dlc_output)
# dlc_output = color_encode(dlc_output, color_values)
# dlc_output = Image.fromarray(np.uint8(dlc_output))

# dlc_output.save('./prediction/flood/test.jpg')
