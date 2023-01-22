import glob
import os
import random

os.chdir("/Users/datle/Desktop/Official_license_plate")
import numpy as np
from sliding_window1 import *
import time
import cv2
import matplotlib.pyplot as plt
from Training_vehicle_detection.setting import win_size
params=load_classifier('vehicle_detect.p')
def run(name, debug=False):
    if type(name)!=str:
        img=name
    else:
        img   = cv2.imread(name, cv2.IMREAD_COLOR)
    #41, 85
    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img= cv2.resize(img, (800,400))
    img1  = img.copy()
    img2  = img.copy()
    start= time.time()
    bbox, bbox_nms= find_car_multi_scale(img,params, win_size)
    if bbox is None and bbox_nms is None:
        return None, None
    end= time.time()
    print(f'time is: {end-start}')
    heatmap=draw_heatmap(bbox, img)
    heatmap_thresh= apply_threshhold(heatmap, thresh=win_size['thresh'])
    bbox_heatmap= get_labeled(heatmap_thresh)
    heatmap_thresh, heatmap= product_heat_and_label_pic(heatmap, heatmap_thresh)
    img2 = draw(img2, bbox_heatmap)
    if debug != False:
        img   =draw(img, bbox)
        img1  =draw(img1, bbox_nms)
        i= np.concatenate((img,img1,img2),axis=0)
        i1= np.concatenate((heatmap, heatmap_thresh), axis=0)
        i1= cv2.resize(i1, (600,300))
        cv2.imshow('i',i)
        cv2.imshow('i1',i1)
        cv2.waitKey(0)
    return img2, bbox_heatmap
def test():
    os.chdir("/Users/datle/Desktop/Official_license_plate/Training_vehicle_detection")
    l=glob.glob("./result/middle_close.jpeg")
    random.shuffle(l)
    for i in l:
        result,bbox= run(i,debug=True)

