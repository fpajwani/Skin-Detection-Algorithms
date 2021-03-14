#reference: https://docs.opencv.org
# Required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time



# get image to be tested from dir
image = cv2.imread(r"C:\Users\mohit\Desktop\Fall 2020\Image Processing\Assignment 2\skin6.jpg")

#defining the lower and upper boundary of skin pixel space in HSV space
Lower_bound = np.array([0, 58, 30], dtype = "uint8")
upper_bound = np.array([33, 255, 255], dtype = "uint8")

start= time.time() #store a timestamp to calculate time taken

cvt_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  #convert the image to HSV colour space 
skin_pixels = cv2.inRange(cvt_image, Lower_bound, upper_bound) #create a mask of pixels that fall under the range of pixels 
skin = cv2.bitwise_and(image, image, mask = skin_pixels)       #Apply the mask to the orginal image so that the non skin pixels
                                                               #are blacked out.

end= time.time()  # record end of algo timestamp
print("Time for Skin_Detection of HSV : ", end - start)  #to calculate time taken by Algorithm

# #******** PLEASE UNCOMMENT THE BELOW PORTION TO SEE THE OUTPUT******
# cv2.imshow('skin detection by hsv',skin)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#write processed image to a file 
cv2.imwrite(r"C:\Users\mohit\Desktop\Fall 2020\Image Processing\Assignment 2\simple_hsv6.jpg",skin,[cv2.IMWRITE_JPEG_QUALITY, 95])
