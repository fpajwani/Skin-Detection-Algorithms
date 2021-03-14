#Machine Learning apporach to skin detection
#Skin Segmentation Data Set from https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation
#reference: https://github.com/rhythm92/Simple-skin-detection

#importing libraries
import numpy as np
import cv2
from sklearn import tree
from sklearn.model_selection import train_test_split
import time

#This function reads data from a text file that contains sample of RGB values of pixels and their label
#whether the pixel is a skin pixel or not.
def ReadData():  

    #reading the dataset as a numpy object from the path
    data = np.genfromtxt(r"C:\Users\mohit\Desktop\Fall 2020\Image Processing\Assignment 2\Skin_NonSkin.txt", dtype=np.int32) 
                                                             # please provide the path to the Skin_Nonskin.txt file here

    #as the dataset has 4 columns: first 3 are RGB values 4th column has labels
    #reading all the label values from data {viz. column 4}
    labels= data[:,3]
    #reading all the RGB values from data {viz. column 1,2,3}
    data= data[:,0:3]

    return data, labels

#in order to avoid illumination dependency during detection 
#this function converts RGB to HSV colour space
def BGR2HSV(bgr):

    # Convert RGB to HSV

    bgr= np.reshape(bgr,(bgr.shape[0],1,3)) #reshape array from [n,3] to [n,1,3]
    hsv= cv2.cvtColor(np.uint8(bgr), cv2.COLOR_BGR2HSV)
    hsv= np.reshape(hsv,(hsv.shape[0],3))   #reshape array from  [n,1,3] to [n,3] 

    return hsv

# This function is a machine learning approach to train the model for skin pixel detection
# in which classification is done using the data and the result is used to predict the test 
# image pixel being a skin pixel or not.

# this function uses 3 parameters
# data: RGB values from dataset to train model
# label: Labels of those pixels
# HSVIndicator: This variable is just used to indicate whether to consider HSV space or not

def TrainTree(data, labels, HSVIndicator): 
    #if HSVIndicator is true convert RGB to HSV
    if(HSVIndicator):
        data= BGR2HSV(data)

    # data segmentation for test, and training dataset
    trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size=0.20, random_state=42)
    
    #************** PLEASE UNCOMMENT THE BELOW BLOCK OF CODE TO SEE THE SIZE OF TRAIN AND TEST DATA SETS*********
    
    # print ("Size of Training Data",trainData.shape)
    # print ("Size of Train Labels",trainLabels.shape)
    # print ("Size of Test Data:",testData.shape)
    # print ("Size of Test Labels:",testLabels.shape)

    # Training the model using decision tree classification with entropy as criteria for split
    clf = tree.DecisionTreeClassifier(criterion='entropy') #declaring an object of tree.DecisionTreeClassifier class 
    clf = clf.fit(trainData, trainLabels) # make descison tree

    print ("Feature Importance",clf.feature_importances_) #print importance of each branch of tree
    print ("Accuracy",clf.score(testData, testLabels))    #checking the accuracy of the fit 

    return clf
# this function can be considered as  main function which reads the image to be tested from a path 
# and calls above functions to complete the task of skin detection and writes the image to a file  
def ApplyToImage(path, HSVIndicator):

    data, labels= ReadData()                        #read data for training
    clf= TrainTree(data, labels, HSVIndicator)      #train the model

    img= cv2.imread(path)  #read image to be tested from the path


    print ("Size of Image:",img.shape)
    data= np.reshape(img,(img.shape[0]*img.shape[1],3)) #convert the image to a 2-d array where each element represent each pixel value 
    print ("Size of Data",data.shape)

    if(HSVIndicator):   
        data= BGR2HSV(data)

    predictedLabels= clf.predict(data) #This method predicts the class label where 1 is for non skin pixel and 2 is for skin pixel
                                       #of the test image pixels using the descision tree that we trained

    #Once all the labels are predicted the array is reshaped into the dimensions of
    #the image. So technically we have masked the test image to their respective labels
    #indicating wheteher that pixel is a skin pixel or not 
    imgLabels= np.reshape(predictedLabels,(img.shape[0],img.shape[1],1))   
#********** PLEASE UNCOMMENT BELOW LINE TO SEE THE PREDICETD LABELS *************
    # print(imgLabels)


    # The Masked image array which contains the label of skin pixels is comverted to white[255,255,255] if it is a skin pixel[2] or 
    # is converted to black[0,0,0] if non skin pixel[1]. likewise image is constructed and written to a file.
    if (HSVIndicator):
        cv2.imwrite(r"C:\Users\mohit\Desktop\Fall 2020\Image Processing\Assignment 2\skin_ml_hsv6.jpg",((-(imgLabels-1)+1)*255))# from [1 2] to [0 255]
        
    else:
        cv2.imwrite(r"C:\Users\mohit\Desktop\Fall 2020\Image Processing\Assignment 2\skin_ml_rgb6.jpg",((-(imgLabels-1)+1)*255))# from [1 2] to [0 255]
        

#---------------------------------------------

start= time.time() #store a timestamp to calculate time taken
# HSV Colorspace is on
ApplyToImage(r"C:\Users\mohit\Desktop\Fall 2020\Image Processing\Assignment 2\skin6.jpg", True)
                                        # please provide the path to the test images here
end= time.time()  # record end of algo timestamp
print("Time for Skin_Detection of HSV : ", end - start)  #to calculate time taken by Algorithm

# start= time.time() #store a timestamp to calculate time taken
# # HSV Colorspace is off
# ApplyToImage(r"C:\Users\mohit\Desktop\Fall 2020\Image Processing\Assignment 2\skin6.jpg", False)
#                                         # please provide the path to the test images here
# end= time.time()  # record end of algo timestamp
# print("Time for Skin_Detection of RGB : ", end - start)  #to calculate time taken by Algorithm.