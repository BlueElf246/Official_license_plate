from pipeline import *
import cv2

cap = cv2.VideoCapture('/Users/datle/Desktop/Official_license_plate/short1.mp4')
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read until video is completed
i=0
fifo = []
while (cap.isOpened()):
    # Capture frame-by-frame
    # if i % 3!=0:
    #     i+=1
    #     continue
    # else:
        ret, frame = cap.read()
        if ret == True:
            run1(frame,i, False)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
            i+=1

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()