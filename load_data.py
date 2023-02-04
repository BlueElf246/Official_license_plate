import glob
import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pickle
def load_dataset(name1, name2):
    car=[]
    for x in name1:
        car+= glob.glob(x)
    non_car=[]
    for y in name2:
        non_car+= glob.glob(y)
    return car, non_car
def color_hist(img,bins):
    img1= np.histogram(img[:,:,0], bins=bins, range=(0,256))
    img2= np.histogram(img[:,:,1], bins=bins, range=(0,256))
    img3= np.histogram(img[:,:,2], bins=bins, range=(0,256))
    return np.concatenate((img1[0], img2[0], img3[0]))
def spatial(img, size):
    img1= cv2.resize(img[:,:,0], size).ravel()
    img2= cv2.resize(img[:,:,1], size).ravel()
    img3= cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((img1, img2, img3))
def get_feature_of_image(img, orient=9, pix_per_cell=8, cell_per_block=2, spatial_size=(16,16),
                         bins=16, feature_vector=True, vis=False ,hog_fea=True, color_fea=True,
                         spatial_fea=True, special=True, color_space='RGB'):
    feature=[]
    img = change_color_space(img, color_space)
    if hog_fea==True:
        h=[]
        if color_space =='gray':
            h.append(hog(img, orient, pixels_per_cell=(pix_per_cell,pix_per_cell), cells_per_block=(cell_per_block,cell_per_block),
                         feature_vector=feature_vector, visualize=vis, transform_sqrt=False))
        else:
            for x in range(3):
                hog_feature= hog(img[:,:,x], orient, pixels_per_cell=(pix_per_cell,pix_per_cell), cells_per_block=(cell_per_block,cell_per_block),
                             feature_vector=feature_vector, visualize=vis, transform_sqrt=False)
                h.append(hog_feature)
        if special==True:
            return h
        feature.append(np.concatenate(h))
    if color_fea==True:
        if color_space =='gray':
            feature.append(np.histogram(img, bins=bins, range=(0,256))[0])
        else:
            color_feature= color_hist(img, bins)
            feature.append(color_feature)
    if spatial_fea==True:
        if color_space == 'gray':
            feature.append(cv2.resize(img, spatial_size).ravel())
        else:
            spatial_feature= spatial(img, spatial_size)
            feature.append(spatial_feature)
    return np.concatenate(feature)
def change_color_space(img,colorspace):
    if colorspace != 'RGB':
        if colorspace == "YCrCb":
            img=cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        if colorspace == 'hls':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        if colorspace == 'yuv':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        if colorspace == 'gray':
            img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
def extract_feature(dataset, color_space, params):
    dataset_feature=[]
    for x in dataset:
        img=cv2.imread(x, cv2.IMREAD_COLOR)
        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized= cv2.resize(img,(params['size_of_window'][0],params['size_of_window'][1]))
        feature=get_feature_of_image(img_resized, orient=params['orient'], pix_per_cell=params['pix_per_cell'], cell_per_block=params['cell_per_block'],hog_fea=params['hog_feat'],
                                     spatial_size=params['spatial_size'], spatial_fea=params['spatial_feat'],bins=params['hist_bins'], color_fea=params['hist_feat'],
                                     feature_vector=True, special=False, color_space=color_space)
        dataset_feature.append(feature)
    return dataset_feature

def combine(car, non_car):
    X= np.vstack((car,non_car)).astype(np.float32)
    y= np.hstack((np.ones(len(car)), np.zeros(len(non_car))))
    return X,y
def normalize(X):
    sc=StandardScaler()
    X_scaled= sc.fit_transform(X)
    return sc, X_scaled
def split(X,y):
    X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test
def train_model(X_train, X_test, y_train, y_test):
    svc=LinearSVC(dual=False, max_iter=1000, penalty='l2')
    svc.fit(X_train, y_train)
    print('Test_score: ', svc.score(X_test,y_test))
    return svc
def save_model(file, svc,sc,params,y):
    os.chdir("/Users/datle/Desktop/Official_license_plate/model")
    with open(file, 'wb') as pfile:
        pickle.dump(
            {'svc': svc,
             'scaler': sc,

             'color_space': params['color_space'],
             'orient': params['orient'],
             'pix_per_cell': params['pix_per_cell'],
             'cell_per_block': params['cell_per_block'],
             'hog_channel': params['hog_channel'],
             'spatial_size': params['spatial_size'],
             'hist_bins': params['hist_bins'],
             'spatial_feat': params['spatial_feat'],
             'hist_feat': params['hist_feat'],
             'hog_feat': params['hog_feat'],
             'test_size': params['test_size'],
             'num_of_feature': svc.coef_.shape[-1],
             'size_of_pic_train': params['size_of_window'],
             'total_of_example': len(y),
             'model_parameter': svc.get_params()
             },
            pfile, pickle.HIGHEST_PROTOCOL)
    os.chdir("/Users/datle/Desktop/Official_license_plate")