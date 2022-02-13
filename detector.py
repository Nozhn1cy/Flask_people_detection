# -*- coding: utf-8 -*-

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

if os.path.isfile('model/detector'):
    model = torch.load('model/detector')
    model.eval()
else:
    import torchvision
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
    model.eval()
    torch.save(model, 'model/detector')

def image_processing(filename):
    img_path = 'images/uploaded/' + str(filename)
    img_numpy = cv2.imread(img_path)[:,:,::-1]
    img = torch.from_numpy(img_numpy.astype('float32')).permute(2,0,1)
    img = img / 255.
    return img, img_numpy

def plot_preds(numpy_img, preds):
  boxes = preds['boxes'].detach().numpy()
  numpy_img = np.array(numpy_img)
  for box in boxes:
        numpy_img = cv2.rectangle(numpy_img,
                                 (box[0],box[1]),
                                 (box[2],box[3]),
                                 255,
                                 3
                                 )
  return numpy_img

def detect(filename):
    DETECT_FOLDER = os.getcwd() + '/images/detected/'

    img, img_numpy = image_processing(filename)
    predictions = model(img[None,...])

    CONF_THRESH = 0.7
    boxes = predictions[0]['boxes'][(predictions[0]['scores'] > CONF_THRESH) & (predictions[0]['labels'] == 1)]
    boxes_dict = {}
    boxes_dict['boxes'] = boxes
    img_with_boxes = plot_preds(img_numpy, boxes_dict)
    plt.imsave(DETECT_FOLDER + str(filename), img_with_boxes.astype('uint8'))





