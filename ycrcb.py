#reference: https://docs.opencv.org
# Required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time



# get image to be tested from dir
image = cv2.imread(r"C:\Users\mohit\Desktop\Fall 2020\Image Processing\Assignment 2\skin3.png")

#defining the lower and upper boundary of skin pixel space in YCrCb space
lower_bound = np.array([0,133,77],np.uint8)
upper_bound = np.array([235,173,127],np.uint8)

start= time.time() #store a timestamp to calculate time taken

cvt_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)  #convert the image to  coYCrCb colour space 
skin_pixels = cv2.inRange(cvt_image, lower_bound, upper_bound) #create a mask of pixels that fall under the range of pixels 
skin = cv2.bitwise_and(image, image, mask = skin_pixels)       #Apply the mask to the orginal image so that the non skin pixels
                                                               #are blacked out.

end= time.time()  # record end of algo timestamp
print("Time for Skin_Detection of YCrCb : ", end - start)  #to calculate time taken by Algorithm

# #******** PLEASE UNCOMMENT THE BELOW PORTION TO SEE THE OUTPUT******
# cv2.imshow('skin detection by hsv',skin)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#write processed image to a file 
cv2.imwrite(r"C:\Users\mohit\Desktop\Fall 2020\Image Processing\Assignment 2\simple_YCrCb3.jpg",skin,[cv2.IMWRITE_JPEG_QUALITY, 95])
