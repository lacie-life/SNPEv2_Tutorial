
import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import onnxruntime as ort
import onnx
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader, letterbox
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel
from yolov7_decode import decode

parser = argparse.ArgumentParser(prog='test.py')
parser.add_argument('--weights', nargs='+', type=str, default='./runs/train/yolov7-custom2/weights/best.pt', help='model.pt path(s)')
parser.add_argument('--data', type=str, default='./LG_sub_dataset/data.yaml', help='*.data path')
parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.05, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
parser.add_argument('--task', default='test', help='train, val, test, speed or study')
parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--verbose', action='store_true', help='report mAP by class')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
parser.add_argument('--project', default='runs/test', help='save to project/name')
parser.add_argument('--name', default='exp', help='save to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
opt = parser.parse_args()
opt.save_json |= opt.data.endswith('coco.yaml')
opt.data = check_file(opt.data)  # check file
print(opt)




batch_size = opt.batch_size
imgsz = opt.img_size
conf_thres = opt.conf_thres
iou_thres = opt.iou_thres
data = opt.data
weights = opt.weights
save_txt = opt.save_txt
save_conf = opt.save_conf
save_json = opt.save_json
save_dir = Path('')
v5_metric = opt.v5_metric
plots = False
cuda = True
training = False
half = False
verbose = True
save_hybrid = opt.save_hybrid
onnx_weights = 're_param_yolov7_LG_dataset_origin.onnx'
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
session = ort.InferenceSession(onnx_weights, providers=providers)
print(session)
set_logging()
device = select_device(opt.device, batch_size=batch_size)

model = attempt_load(weights, map_location=device)  # load FP32 model
gs = max(int(model.stride.max()), 32)  # grid size (max stride)
imgsz = check_img_size(imgsz, s=gs)  # check img_size

model.eval()
if isinstance(data, str):
    is_coco = data.endswith('coco.yaml')
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
check_dataset(data)  # check


nc = int(data['nc'])  # number of classes
iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
niou = iouv.numel()

if not training:
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
    dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                    prefix=colorstr(f'{task}: '))[0]
    
seen = 0
confusion_matrix = ConfusionMatrix(nc=nc)
names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}

s = ('%20s' + '%12s' * 7) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.75' , 'mAP@.5:.95')
p, r, f1, mp, mr, map50, map75, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
loss = torch.zeros(3, device=device)
jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

inname = [i.name for i in session.get_inputs()]
outname = [i.name for i in session.get_outputs()]
# 4,10,  6,16,  11,9,  10,22,  21,11,  35,16,  26,31,  45,27,  78,64

anchor = [torch.tensor([[[[[4., 10.]]],


         [[[6., 16.]]],


         [[[11., 9.]]]]]).to(device),
torch.tensor([[[[[ 10.,  22.]]],


         [[[ 21.,  11.]]],


         [[[ 35., 16.]]]]]).to(device),
torch.tensor([[[[[26., 31.]]],


         [[[45., 27.]]],


         [[[78., 64.]]]]]).to(device)]

stride_ = torch.tensor([8.,16.,32.],dtype = torch.float32).to(device)
grid = []
grid_ = [52,26,13]

for i in range(3):
    ny = nx = grid_[i]
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    grid.append(torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float().to(device))

for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):

    img = img.to(device, non_blocking=True)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0    
    targets = targets.to(device)
    nb, _, height, width = img.shape  # batch size, channels, height, width    
    img = img.cpu().numpy()
    inp = {inname[0]:img}
    outputs = session.run(outname, inp) #outputs shape(1 3 52 52 8 , 1 3 26 26 8 , 1 3 13 13 8)

    output_1 = torch.tensor(outputs[0],dtype = torch.float32).contiguous().to(device)
    output_2 = torch.tensor(outputs[1],dtype = torch.float32).contiguous().to(device)
    output_3 = torch.tensor(outputs[2],dtype = torch.float32).contiguous().to(device)

    output = [output_1,output_2,output_3]
    out,_ = decode(output,stride = stride_, anchor = anchor, grid = grid, nc = nc , bs = nb)

    targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
    lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
    
    out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)

    for si, pred in enumerate(out):
        labels = targets[targets[:, 0] == si, 1:]
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class
        path = Path(paths[si])
        seen += 1

        if len(pred) == 0:
            if nl:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue

        # Predictions
        predn = pred.clone()
        if save_txt:
            gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
            for *xyxy, conf, cls in predn.tolist():
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')


        correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
        if nl:
            detected = []  # target indices
            tcls_tensor = labels[:, 0]

            # target boxes
            tbox = xywh2xyxy(labels[:, 1:5])

            # Per target class
            for cls in torch.unique(tcls_tensor):
                ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                # Search for detections
                if pi.shape[0]:
                    # Prediction to target ious
                    ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                    # Append detections
                    detected_set = set()
                    for j in (ious > iouv[0]).nonzero(as_tuple=False):
                        d = ti[i[j]]  # detected target
                        if d.item() not in detected_set:
                            detected_set.add(d.item())
                            detected.append(d)
                            correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                            if len(detected) == nl:  # all targets already located in image
                                break

        # Append statistics (correct, conf, pcls, tcls)
        stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
if len(stats) and stats[0].any():
    p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, v5_metric=v5_metric, save_dir=save_dir, names=names)
    ap50, ap75, ap = ap[:, 0], ap[:, 5], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    mp, mr, map50, map75, map = p.mean(), r.mean(), ap50.mean(), ap75.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
else:
    nt = torch.zeros(1)

# Print results
pf = '%20s' + '%12i' * 2 + '%12.3g' * 5  # print format
print(pf % ('all', seen, nt.sum(), mp, mr, map50, map75, map))

# Print results per class
if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
    for i, c in enumerate(ap_class):
        print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap75[i], ap[i]))
