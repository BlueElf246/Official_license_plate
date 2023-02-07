import os
os.chdir("/Users/datle")
from load_data import *
from Training_vehicle_detection.setting import params, win_size
import numpy as np
import random
name2=["./Desktop/Official_license_plate/Training_vehicle_detection/dataset/non-vehicles/GTI_Far/*.png"]
name2.append("./Desktop/Official_license_plate/Training_vehicle_detection/dataset/non-vehicles/GTI_Left/*.png")
name2.append("./Desktop/Official_license_plate/Training_vehicle_detection/dataset/non-vehicles/GTI_MiddleClose/*.png")
name2.append("./Desktop/Official_license_plate/Training_vehicle_detection/dataset/non-vehicles/GTI_Right/*.png")

name1=["./Desktop/Official_license_plate/Training_vehicle_detection/dataset/vehicles/GTI_Far/*.png"]
name1.append("./Desktop/Official_license_plate/Training_vehicle_detection/dataset/vehicles/GTI_Left/*.png")
name1.append("./Desktop/Official_license_plate/Training_vehicle_detection/dataset/vehicles/GTI_MiddleClose/*.png")
name1.append("./Desktop/Official_license_plate/Training_vehicle_detection/dataset/vehicles/GTI_Right/*.png")
name1.append("./Desktop/Official_license_plate/Training_vehicle_detection/dataset/vehicles/KITTI_extracted/*.png")
car, non_car= load_dataset(name1, name2)
name='svc_nu'
print(len(car), len(non_car))
# car=car[:3]
# non_car=non_car[:3]
car_feature=extract_feature(car, params['color_space'], params)
non_car_feature= extract_feature(non_car, params['color_space'], params)
X,y= combine(car_feature, non_car_feature)
if name =='adaboost' or name =='xgboost':
    sc, X_scaled= None, X
else:
    sc, X_scaled = normalize(X)
X_train, X_test, y_train, y_test= split(X_scaled, y)
print('start to train model')
model= train_model(X_train, X_test, y_train, y_test, model=name)
save_model(f'vehicle_detect_{name}_50.p', model, sc, params=params,y=y)

#vehicle_detect_svc_50.p: best