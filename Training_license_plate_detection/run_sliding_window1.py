import os
os.chdir("/Users/datle/Desktop/Official_license_plate")
import numpy as np
from sliding_window1 import *
import time
import cv2
import matplotlib.pyplot as plt
from Training_license_plate_detection.setting import win_size
import glob
params=load_classifier('lp_detect.p')
def run(name, debug=False, use_nms=False):
    if type(name)!=str:
        img=name
    else:
        img   = cv2.imread(name, cv2.IMREAD_COLOR)
    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2  = img.copy()
    start= time.time()
    bbox, bbox_nms= find_car_multi_scale(img,params, win_size)
    if bbox is None and bbox_nms is None:
        return None, None
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
    cv2.imshow('result1', img2)
    return img2, bbox_nms
def test():
    os.chdir("/Users/datle/Desktop/Official_license_plate")
    l=glob.glob("./image_vehicle/*.jpg")
    for x in l:
        result,bbox= run(x,debug=True)