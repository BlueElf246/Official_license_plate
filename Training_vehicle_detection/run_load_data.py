import os
os.chdir("/Users/datle/Desktop/Official_license_plate")
from load_data import *
from Training_vehicle_detection.setting import params
import numpy as np
import random
name1=["./Training_vehicle_detection/dataset/cars_train/*.jpg"]
name2=["./Training_license_plate_detection/dataset/natural_images/airplane/*.jpg"]
# name2.append("./Training_license_plate_detection/dataset/natural_images/car/*.jpg")
name2.append("./Training_license_plate_detection/dataset/natural_images/cat/*.jpg")
name2.append("./Training_license_plate_detection/dataset/natural_images/dog/*.jpg")
name2.append("./Training_license_plate_detection/dataset/natural_images/flower/*.jpg")
name2.append("./Training_license_plate_detection/dataset/natural_images/fruit/*.jpg")
name2.append("./Training_license_plate_detection/dataset/natural_images/motorbike/*.jpg")
name2.append("./Training_license_plate_detection/dataset/natural_images/person/*.jpg")

name2.append("./Training_license_plate_detection/dataset/non_vehicles/GTI_Far/*.png")
name2.append("./Training_license_plate_detection/dataset/non_vehicles/GTI_Left/*.png")
name2.append("./Training_license_plate_detection/dataset/non_vehicles/GTI_MiddleClose/*.png")
name2.append("./Training_license_plate_detection/dataset/non_vehicles/GTI_Right/*.png")

car, non_car= load_dataset(name1, name2)
random.shuffle(car)
random.shuffle(non_car)
# car=car[:300]
# non_car=non_car[:3]
car_feature=extract_feature(car, params['color_space'], params)
non_car_feature= extract_feature(non_car, params['color_space'], params)
X,y= combine(car_feature, non_car_feature)
sc, X_scaled= normalize(X)
X_train, X_test, y_train, y_test= split(X_scaled, y)
model= train_model(X_train, X_test, y_train, y_test)
save_model('vehicle_detect.p', model, sc, params=params,y=y)