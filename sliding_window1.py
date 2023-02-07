import os

from load_data import get_feature_of_image, change_color_space
import cv2
import numpy as np
import pickle
from imutils.object_detection import  non_max_suppression
from scipy.ndimage import label
def load_classifier(name):
    os.chdir("/Users/datle/Desktop/Official_license_plate/model")
    d= pickle.load(open(name, 'rb'))
    return d
def sliding_window(img,params,scale, y_start_stop=[None, None], cell_per_step=8):
    bbox=[]
    if y_start_stop[0] == None or y_start_stop[0] > img.shape[0]:
        y_start_stop[0]=0
    if y_start_stop[1]== None  or y_start_stop[1] > img.shape[0]:
        y_start_stop[1]= img.shape[0]

    img=img[y_start_stop[0]:y_start_stop[1],:,:]
    win_x, win_y, channel= params['size_of_pic_train']
    pix_per_cell= params['pix_per_cell']
    svc= params['svc']
    scaler= params['scaler']
    shape_img= img.shape
    if scale !=1:
        img=cv2.resize(img,(np.int32(shape_img[1]/scale), np.int32(shape_img[0]/scale)))
        shape_img = img.shape
    number_of_cell_in_x= (shape_img[1]//pix_per_cell)-1
    number_of_cell_in_y= (shape_img[0]//pix_per_cell)-1

    number_of_cell_per_window_x= (win_x//pix_per_cell)-1
    number_of_cell_per_window_y= (win_y//pix_per_cell)-1

    number_of_window_in_x= (number_of_cell_in_x-number_of_cell_per_window_x)//cell_per_step
    number_of_window_in_y= (number_of_cell_in_y-number_of_cell_per_window_y)//cell_per_step


    h= get_feature_of_image(img, orient=params['orient'], pix_per_cell=params['pix_per_cell'], cell_per_block=params['cell_per_block'],hog_fea=params['hog_feat'],
                                     spatial_size=params['spatial_size'], spatial_fea=params['spatial_feat'],bins=params['hist_bins'], color_fea=params['hist_feat'],
                                feature_vector=False, special=True, color_space=params['color_space'])
    ch1=[]
    ch2=[]
    ch3=[]
    if params['color_space']=='gray':
        ch1=h[0]
    else:
        ch1=h[0]
        ch2=h[1]
        ch3=h[2]

    for y in range(number_of_window_in_y+1):
        for x in range(number_of_window_in_x+1):
            x_pos= x*cell_per_step
            y_pos= y*cell_per_step
            hog_fea2=[]
            hog_fea3=[]
            if params['color_space'] == 'gray':
                hog_fea1= ch1[y_pos:y_pos+number_of_cell_per_window_y, x_pos:x_pos+number_of_cell_per_window_x].ravel()
            else:
                hog_fea1 = ch1[y_pos:y_pos + number_of_cell_per_window_y,x_pos:x_pos + number_of_cell_per_window_x].ravel()
                hog_fea2= ch2[y_pos:y_pos+number_of_cell_per_window_y, x_pos:x_pos+number_of_cell_per_window_x].ravel()
                hog_fea3= ch3[y_pos:y_pos + number_of_cell_per_window_y, x_pos:x_pos + number_of_cell_per_window_x].ravel()
            hog_f= np.hstack((hog_fea1,hog_fea2,hog_fea3))
            x_top= x_pos * pix_per_cell
            y_top= y_pos * pix_per_cell


            img_crop= cv2.resize(img[y_top:y_top+win_y,x_top:x_top+win_x],(win_x,win_y))
            spatial_feature=[]
            color_feature=[]
            if params['spatial_feat'] == True:
                spatial_feature = get_feature_of_image(img_crop, hog_fea=False, spatial_fea=params['spatial_feat'], spatial_size=params['spatial_size'], color_fea=False)
            if params['hist_feat'] == True:
                color_feature   = get_feature_of_image(img_crop, hog_fea=False, spatial_fea=False, bins=params['hist_bins'], color_fea=params['hist_feat'])
            feature= np.concatenate((hog_f, color_feature, spatial_feature))
            if scaler is None:
                scaled_feature=np.array(feature).reshape(1,-1)
            else:
                scaled_feature=scaler.transform(np.array(feature).reshape(1,-1))
            prediction= svc.predict(scaled_feature)
            if prediction ==1:
                xbox_left = np.int32(x_top * scale)
                ytop_draw = np.int32(y_top * scale)
                win_draw_x = np.int32(win_x * scale)
                win_draw_y = np.int32(win_y * scale)
                bbox.append([xbox_left, ytop_draw + y_start_stop[0],xbox_left + win_draw_x, ytop_draw + win_draw_y + y_start_stop[0]])
    return bbox
def find_car_multi_scale(img,params, win_size):
    bboxes=[]
    # print('number of scale use:', len(win_size['use_scale']))
    win_scale=win_size['use_scale']
    y_start_stop= win_size['y_start_stop']
    for x in range(win_size['length']):
        if x in win_scale:
            scale_0=win_size[f'scale_{x}'][2]
            bbox=sliding_window(img, params=params, y_start_stop= y_start_stop, cell_per_step=2, scale=scale_0)
            if len(bbox)==0:
                continue
            bboxes.append(bbox)
    if len(bboxes) ==0:
        return None,None
    bboxes= np.concatenate(bboxes)
    return bboxes,non_max_suppression(np.array(bboxes),probs=None, overlapThresh=win_size['overlap_thresh'])
def draw(img,box):
    for x in box:
        cv2.rectangle(img, (x[0],x[1]), (x[2],x[3]), (0,0,255), 2)
    return img
def draw_heatmap(bbox,img):
    img_new= np.zeros_like(img)
    for box in bbox:
        img_new[box[1]:box[3],box[0]:box[2]]+=1
    print('threshhold: ',img_new[...,1].max())
    return img_new
def apply_threshhold(heatmap,thresh=3):
    heatmap=np.copy(heatmap)
    heatmap[heatmap< thresh]=0
    heatmap= np.clip(heatmap,0,254)
    return heatmap
def get_labeled(heatmap_thresh):
    labels= label(heatmap_thresh)
    bboxes = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ([np.min(nonzerox), np.min(nonzeroy), np.max(nonzerox), np.max(nonzeroy)])
        bboxes.append(bbox)
    # Return list of bounding boxes
    return bboxes

def product_heat_and_label_pic(heatmap, labels):
    # prepare RGB heatmap image from float32 heatmap channel
    img_heatmap = (np.copy(heatmap) / np.max(heatmap) * 255.).astype(np.uint8)
    img_heatmap = cv2.applyColorMap(img_heatmap, colormap=cv2.COLORMAP_JET)
    img_heatmap = cv2.cvtColor(img_heatmap, cv2.COLOR_BGR2RGB)

    # prepare RGB labels image from float32 labels channel
    img_labels = (np.copy(labels) / np.max(labels) * 255.).astype(np.uint8)
    img_labels = cv2.applyColorMap(img_labels, colormap=cv2.COLORMAP_HOT)
    img_labels = cv2.cvtColor(img_labels, cv2.COLOR_BGR2RGB)

    return img_labels, img_heatmap

def get_prediction_of_image(params,img):
    img= cv2.resize(img,(params['size_of_pic_train'][0],params['size_of_pic_train'][1]))
    svc= params['svc']
    scaler=params['scaler']
    feature = get_feature_of_image(img, orient=params['orient'], pix_per_cell=params['pix_per_cell'], cell_per_block=params['cell_per_block'],hog_fea=params['hog_feat'],
                                     spatial_size=params['spatial_size'], spatial_fea=params['spatial_feat'],bins=params['hist_bins'], color_fea=params['hist_feat'],
                                feature_vector=True, special=False, color_space=params['color_space'])
    scaled_feature= scaler.transform(np.array(feature).reshape(1,-1))
    if svc.predict(scaled_feature)==1:
        return True
