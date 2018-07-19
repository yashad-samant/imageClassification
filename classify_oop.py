# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 23:50:52 2018

@author: Yashad
"""

# Import necessary libraries
# OpenCV for image extraction, preprocessing, colorspace conversion
# numpy for mandatory matrix operations
# sklearn for Support Vector Machines algorithm
import cv2
import matplotlib.pyplot as plt
import glob 
import numpy as np
from sklearn import svm


# Class generates a trained SVM model on 80% dataset
# It consists of the mandatory diagnose function, the only user input is the path of the image and the output is the class
# Class consists of six functions --
## 1. importImages, 2. Normalization, 3. Resize, 4. dataProcessing, 5. SVM, 6. diagnose

##------------------------------------------------Object Code--------------------------------------------------------------


class Classification():

# Global variable defined in class    
    imageList = []
    
# Import images from the folder & save it in a array
# Functions normalization and resize are also called in this function. Thus, as each image is imported from
# the folder, they are normalized by making it zero mean and dividing by standard deviation
# All the images are resized to 300x300 resolution.
    
    def importImages(self, files):
        self.imageList = []
        for f1 in files:
            img = cv2.imread(f1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = self.normalization(img)
            img = self.resize(img)
            self.imageList.append(img)
        self.imageList = np.asarray(self.imageList)
        return self.imageList
        
# Standardize the image by subtracting the mean and dividing by the standard deviation
    def normalization(self,img):
        m = np.mean(img)
        s = np.std(img)
        img = (img-m)/s
        return img

# All images are of varying resolution. For consistency, images are affine transformed to 300x300.    
    def resize(self,img):
        img = cv2.resize(img, (300,300), interpolation = cv2.INTER_CUBIC)
        img = img.reshape(90000,)
        return img

# In this function, we have called importImages function and stored pathology A and B folders in A & B
# numpy arrays respectively. Then, A & B are vertically stacked into X and randomized so that SVM kernel is not trained
# specifically for A first or vice-versa. It might generate errors due to over-fitting.
# Y are basically the classes for each image.
    
    def dataProcessing(self):
        files_A = glob.glob('pathology A/*.png')
        files_B = glob.glob('pathology B/*.png')
        A = self.importImages(files_A)
        B = self.importImages(files_B)
        
        
        A = np.vstack(A)
        B = np.vstack(B)
        
        r = np.arange(20)
        np.random.shuffle(r)
        
        X = np.vstack((A,B))
        X = X[r]
        Y = np.asarray((['A']*10)+(['B']*10))
        Y = Y[r]
        return X, Y
    
# Data was divided into training and test data in the ratio 4:1. 
# SVMs were used for classification because of the non-linearity over LDA.
# Polynomial with degree 2 kernel was used for training. It provided better results than RBF and Linear kernel
    def SVM(self):
        X, Y = self.dataProcessing()
    # Classification algorithm
        X_train  = X[0:16]
        X_test = X[16:20]
        Y_train  = Y[0:16]
        Y_test = Y[16:20]
        target_names=['A','B']
        
        estimator_poly = svm.SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape='ovr', degree=2, gamma=0.0001, kernel='poly',
            max_iter=-1, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=True)
        estimator_poly.fit(X_train,Y_train)
        return estimator_poly

    def diagnose(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = self.normalization(image)
        image = self.resize(image)
        estimator_poly = self.SVM()
        return estimator_poly.predict(image.reshape(1,-1))
    
    

# Object call
Classify = Classification()

# User Input
image = cv2.imread('pathology B/B9.png')

# Predicted output
print("      The image belongs to:", Classify.diagnose(image))
      




    


