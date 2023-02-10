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
params=load_classifier('vehicle_detect_svc_50.p')
def filter_plate(bbox):
    for i,x in enumerate(bbox):
        # if (x[3]- x[1]) <15 and (x[2]- x[0]) <50:
        #     continue
        width= x[2]- x[0]
        height= x[3]- x[1]
        ratio= np.round_(width/height)
        if ratio in (1,):
            if (width >60) and (width<100) and (height >60) and (height <100):
                bbox[i]=x
    return bbox
def run(name, debug=False, use_nms=False):
    if type(name)!=str:
        img=name
    else:
        img = cv2.imread(name, cv2.IMREAD_COLOR)
    # img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img= cv2.resize(img, (1000,500))
    img2  = img.copy()
    start= time.time()
    bbox, bbox_nms= find_car_multi_scale(img,params, win_size)
    if bbox is None and bbox_nms is None:
        return img2, None
    end= time.time()
    print(f'time is: {end-start}')
    if use_nms==False:
        heatmap=draw_heatmap(bbox, img)
        heatmap_thresh= apply_threshhold(heatmap, thresh=win_size['thresh'])
        bbox_heatmap= get_labeled(heatmap_thresh)
        bbox_heatmap=filter_plate(bbox_heatmap)
        img2 = draw(img2, bbox_heatmap)
    else:
        img2 = draw(img2, bbox_nms)
    if debug != False:
        if use_nms==True:
            img = draw(img, bbox)
            cv2.imshow('i', img)
        else:
            img1=img.copy()
            heatmap_thresh, heatmap = product_heat_and_label_pic(heatmap, heatmap_thresh)
            img1  =draw(img1, bbox_nms)
            i= np.concatenate((img,img1,img2),axis=0)
            i1= np.concatenate((heatmap, heatmap_thresh), axis=0)
            i1= cv2.resize(i1, (600,300))
            cv2.imshow('i',i)
            cv2.imshow('i1',i1)
    # cv2.imshow('result', img2)
    return img2, bbox_nms
def test():
    os.chdir("/Users/datle/Desktop/Official_license_plate")
    l=glob.glob("./Training_vehicle_detection/result/middle_close.jpeg")
    random.shuffle(l)
    for i in l:
        result,bbox= run(i,debug=True, use_nms=False)
        cv2.imshow('r', result)
        cv2.waitKey(0)

test()