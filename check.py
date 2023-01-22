from Training_license_plate_detection.run_sliding_window1 import params as params_plate
from Training_vehicle_detection.run_sliding_window1 import params as params_vehicle
from sliding_window1 import get_prediction_of_image
import cv2

img= cv2.imread('/Users/datle/Desktop/Official_license_plate/image_vehicle/0_1.jpg', cv2.IMREAD_COLOR)
img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(get_prediction_of_image(params_vehicle, img))
