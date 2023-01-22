import os
os.chdir("/Users/datle/Desktop/Official_license_plate/Training_vehicle_detection/dataset/cars_train1")
import glob
import random
file_delete= glob.glob("./*.jpg")
print(len(file_delete))
random.shuffle(file_delete)
file_delete=file_delete[:4000]
for x in file_delete:
    os.remove(x)
