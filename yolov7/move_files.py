import os
import shutil
import natsort

f = open("/home/jiwon/3d_object_detection/Complex-YOLOv4-Pytorch/dataset/A9_dataset/ImageSets/new_test_north_south.txt", "r")
lines = f.readlines()
path = "/home/jiwon/3d_object_detection/Complex-YOLOv4-Pytorch/dataset/A9_dataset/training/image_north_south_1"
dir_path = "/home/jiwon/3d_object_detection/yolov7/a9_dataset_image_north_south"
for line in lines:
    line = line.strip()  # 줄 끝의 줄 바꿈 문자를 제거한다.
    image_path = os.path.join(path,line+".jpg")
    shutil.copyfile(image_path, os.path.join(dir_path,line+".jpg"))