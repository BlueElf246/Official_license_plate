import os
os.chdir("/Users/datle")
from load_data import *
from Training_vehicle_detection.setting import params, win_size
import numpy as np
import random
# name1=["./Desktop/Official_license_plate/Training_vehicle_detection/dataset/cars_train/*.jpg"]
# name2=["./Desktop/Official_license_plate/Training_license_plate_detection/dataset/natural_images/airplane/*.jpg"]
# name2.append("./Training_license_plate_detection/dataset/natural_images/car/*.jpg")

name2=["./Downloads/data/non-vehicles/*.png"]
name1=["./Downloads/data/vehicles/*.png"]
name1.append("./Downloads/data/positive/img/*.jpg")
name2.append("./Downloads/data/negative/img/*.jpg")
# name2=["./Downloads/data/negative/img/*.jpg"]
# name2.append("./Desktop/Official_license_plate/Training_license_plate_detection/dataset/non_vehicles/GTI_Far/*.png")
# name2.append("./Desktop/Official_license_plate/Training_license_plate_detection/dataset/non_vehicles/GTI_Left/*.png")
# name2.append("./Desktop/Official_license_plate/Training_license_plate_detection/dataset/non_vehicles/GTI_MiddleClose/*.png")
# name2.append("./Desktop/Official_license_plate/Training_license_plate_detection/dataset/non_vehicles/GTI_Right/*.png")

# name2.append("./Desktop/Official_license_plate/Training_license_plate_detection/dataset/natural_images/cat/*.jpg")
# name2.append("./Desktop/Official_license_plate/Training_license_plate_detection/dataset/natural_images/dog/*.jpg")
# name2.append("./Desktop/Official_license_plate/Training_license_plate_detection/dataset/natural_images/flower/*.jpg")
# name2.append("./Desktop/Official_license_plate/Training_license_plate_detection/dataset/natural_images/fruit/*.jpg")
# name2.append("./Desktop/Official_license_plate/Training_license_plate_detection/dataset/natural_images/motorbike/*.jpg")
# name2.append("./Desktop/Official_license_plate/Training_license_plate_detection/dataset/natural_images/person/*.jpg")
# name2.append("./Desktop/Official_license_plate/Training_license_plate_detection/dataset/natural_images/airplane/*.jpg")
car, non_car= load_dataset(name1, name2)
# car=sorted(car, key=lambda car: car.split('/')[-1])
# car_new=[]
# i=0
# d={'far':[], 'left':[], 'middle':[], 'right':[]}
# for y,x in enumerate(car):
#     last=x.split('/')[-1]
#     for k in d:
#         if k in last:
#             d[k].append(x)
# for x in d:
#     car_new+=d[x][:50]
name='svc'
print(len(car), len(non_car))
random.shuffle(car)
random.shuffle(non_car)
car=car[:100]
#300
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