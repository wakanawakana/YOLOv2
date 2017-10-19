#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import cv2
import numpy as np
import chainer
import glob
import os
from chainer import serializers, optimizers, Variable, cuda
import chainer.functions as F
from yolov2 import *
from lib.utils import *
from lib.image_generator import *


def generate_data(path):
    x = []
    t = []
    ground_truths = []
    img_orig = cv2.imread(path)
    #img = reshape_to_yolo_size(img_orig)
    img = cv2.resize(img_orig, (416, 416))
    w_fine = (float((img_orig.shape[1] / 32) * 32) / float(img_orig.shape[1]) )
    h_fine = (float((img_orig.shape[0] / 32) * 32) / float(img_orig.shape[0]) )
       
    fimg = img_orig.astype(np.float)
    input_height, input_width, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, dtype=np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    x.append(img)


    ground_truths.append({
        "x": 0.67084,
        "y": 0.610435,
        "w": 0.227971 * w_fine,
        "h": 0.647691 * h_fine,
        "label": 0,
    }) 

    t.append(ground_truths)
    x = np.array(x)
    return x, t, fimg
# hyper parameters
train_sizes = [320, 352, 384, 416, 448]
item_path = "./items"
background_path = "./backgrounds"
initial_weight_file = "./yolov2_darknet.model"
weight_file = "./yolov2_darknet.model"
backup_path = "backup"
backup_file = "%s/backup.model" % (backup_path)
batch_size = 16
max_batches = 1
learning_rate = 1e-5
learning_schedules = { 
    "0"    : 1e-5,
    "500"  : 1e-4,
    "10000": 1e-5,
    "20000": 1e-6 
}

lr_decay_power = 4
momentum = 0.9
weight_decay = 0.005
n_classes = 80
n_boxes = 5

# load image generator
print("loading image generator...")
generator = ImageGenerator(item_path, background_path)

# load model
print("loading initial model...")
yolov2 = YOLOv2(n_classes=n_classes, n_boxes=n_boxes)

model = YOLOv2Predictor(yolov2)
serializers.load_hdf5(weight_file, yolov2)
model.unstable_seen = 0
model.thresh = 0.5

model.predictor.train = True
model.predictor.finetune = False
cuda.get_device(0).use()
#model.to_gpu()

optimizer = optimizers.MomentumSGD(lr=learning_rate, momentum=momentum)
optimizer.use_cleargrads()
optimizer.setup(model)
#optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

# start to train
print("start training")
for batch in range(max_batches):
    if str(batch) in learning_schedules:
        optimizer.lr = learning_schedules[str(batch)]

    # generate sample
    x, t, fimg = generate_data('./data/people.png')
    x = Variable(x)
    #x.to_gpu()

    # forward
    #yolov2.zerograds()
    loss = model.gcam(x, t, target=18)
    print("batch: %d learning rate: %f loss: %f" % (batch, optimizer.lr, loss.data))

    # backward and optimize
    optimizer.zero_grads()
    loss.backward(retain_grad=True)
    weights = np.mean(yolov2.gcamout.grad, axis=(2, 3))
    #weights = abs(weights)
    gcam = np.tensordot(weights[0], yolov2.gcamout.data[0], axes=(0, 0))
    gcam = (gcam > 0) * gcam / gcam.max()
    gcam = (gcam * 255)
    gcam = cv2.resize(np.uint8(gcam), (fimg.shape[1], fimg.shape[0]))
    heatmap = cv2.applyColorMap(gcam, cv2.COLORMAP_JET)
    gcam = fimg + np.float32(heatmap)
    gcam = 255 * gcam / gcam.max()
    
    for truth_index in range(len(t[0])):
        left = int(t[0][truth_index]["x"]*gcam.shape[1] - t[0][truth_index]["w"]*gcam.shape[1]/2 ) 
        top = int(t[0][truth_index]["y"]*gcam.shape[0] - t[0][truth_index]["h"]*gcam.shape[0]/2 ) 
        right = int(t[0][truth_index]["x"]*gcam.shape[1] + t[0][truth_index]["w"]*gcam.shape[1]/2 ) 
        bottom = int(t[0][truth_index]["y"]*gcam.shape[0] + t[0][truth_index]["h"]*gcam.shape[0]/2 ) 
        cv2.rectangle(
            gcam,
            (left, top), (right, bottom),
            (0, 128, 255),
            1
        )
    cv2.imwrite('gcam.png', gcam)


