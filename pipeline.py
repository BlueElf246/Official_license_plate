import glob
import os
import cv2
import numpy as np
os.chdir("/Users/datle/Desktop/Official_license_plate")
from Training_license_plate_detection.run_sliding_window1 import run as detect_plate
from Training_vehicle_detection.run_sliding_window1 import run as detect_vehicle
from Training_license_plate_detection.run_sliding_window1 import params as params_plate
from Training_vehicle_detection.run_sliding_window1 import params as params_vehicle
from sliding_window1 import get_prediction_of_image
def show(img):
    cv2.imshow('i', img)
    cv2.waitKey(0)
def filter_vehicle(img, bbox, use_nms):
    imgs=[]
    for x in bbox:
        if use_nms!=True:
            width = x[2] - x[0]
            height = x[3] - x[1]
            print(width, height)
            ratio = np.round_(width / height)
            if ratio in (1,2,3):
                if (width>90 and width<400) and (height>90 and height<300):
                    img_crop=img[x[1]+2:x[3]-2,x[0]+2:x[2]-2,:]
                    # if get_prediction_of_image(params_vehicle, img_crop) == True :
                    imgs.append(img_crop)
        else:
            img_crop = img[x[1] + 2:x[3] - 2, x[0] + 2:x[2] - 2, :]
            # if get_prediction_of_image(params_vehicle, img_crop) == True :
            imgs.append(img_crop)
    return imgs
def filter_plate(img, bbox, use_nms):
    imgs=[]
    for x in bbox:
        if use_nms != True:
            # if (x[3]- x[1]) <15 and (x[2]- x[0]) <50:
            #     continue
            width= x[2]- x[0]
            height= x[3]- x[1]
            ratio= np.round_(width/height)
            if ratio in (3,4,5):
                if (width >60 and width<300) and (height >10 and height <100):
                    img_crop= img[x[1]+2:x[3]-2,x[0]+2:x[2]-2,:]
                    # if get_prediction_of_image(params_plate, img_crop) == True :
                    imgs.append(img_crop)
        else:
            img_crop = img[x[1] + 2:x[3] - 2, x[0] + 2:x[2] - 2, :]

            # if get_prediction_of_image(params_plate, img_crop) == True :
            imgs.append(img_crop)

    return imgs
def run():
    img_input = sorted(glob.glob("/Users/datle/Desktop/Official_license_plate/images/*.*"))
    for x,img in enumerate(img_input):
        result, bbox= detect_vehicle(img, debug=True)
        if result is None and bbox is None:
            continue
        imgs=filter_vehicle(result, bbox, True)
        if len(imgs)==0:
            continue
        for y,img in enumerate(imgs):
            if img.shape[0]==0 or img.shape[1]==0:
                continue
            cv2.imwrite(f'/Users/datle/Desktop/Official_license_plate/image_vehicle/{x}_{y}.jpg', img)

    img_vehicle= sorted(glob.glob("/Users/datle/Desktop/Official_license_plate/image_vehicle/*.jpg"))
    for x,img in enumerate(img_vehicle):
        x1=img.split('/')[-1][:3]
        result, bbox= detect_plate(img, debug=True)
        # print('boxes:',number_box)
        # if number_box <100:
        #     continue
        if result is None and bbox is None:
            x+=1
            continue
        imgs=filter_plate(result, bbox, True)
        if len(imgs)==0:
            x+=1
            continue
        for y,img1 in enumerate(imgs):
            if img1.shape[0] == 0 or img1.shape[1] == 0:
                y+=1
                continue
            cv2.imwrite(f'/Users/datle/Desktop/Official_license_plate/image_plate/{x1}_{x}_{y}.jpg', img1)


def run1(img,x,debug, use_nms):
    result, bbox = detect_vehicle(img, debug=debug, use_nms=use_nms)
    cv2.imshow('result', result)
def run2(img,x,debug, use_nms):
    pass
def change_green(img):
    light_green= (25, 52, 72)
    dark_green= (102, 255, 255)
    img_src=img.copy()
    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, light_green, dark_green)
    result= cv2.bitwise_and(img_src, img_src, mask=mask)
    cv2.imshow('reuslt', result)
    cv2.waitKey(0)
img=cv2.imread("/Users/datle/Desktop/Official_license_plate/images/Screen Shot 2023-02-03 at 14.12.24.png", cv2.IMREAD_COLOR)
change_green(img)






