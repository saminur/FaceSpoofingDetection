# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 22:25:50 2020

@author: sami
"""


import numpy as np
import cv2
import os
from scipy import ndimage
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# takes all images and convert them to grayscale. 
# return a dictionary that holds all images category by category. 
def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        category = []
        path = folder + "/" + filename
        for cat in os.listdir(path):
            img = cv2.imread(path + "/" + cat,0)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img is not None:
                category.append(img)
        images[filename] = category
    return images
train_path = "train"
test_path = "test"
print("image load start. ")
images = load_images_from_folder(train_path)  # take all images category by category 
test = load_images_from_folder(test_path) # take test images 
print("image load Finish")

def orb_features(images):
    orb_vectors = {}
    descriptor_list = []
    orb = cv2.ORB_create()
    for key,value in images.items():
        features = []
        for img in value:
            kp, des = orb.detectAndCompute(img,None)
            if des is None:
                continue
            descriptor_list.extend(des)
            features.append(des)
        orb_vectors[key] = features
    return [descriptor_list, orb_vectors]

print("image Feature extractor start")
orbs = orb_features(images) 
# Takes the descriptor list which is unordered one
descriptor_list = orbs[0] 
# Takes the sift features that is seperated class by class for train data
all_bovw_feature = orbs[1] 
# Takes the sift features that is seperated class by class for test data
test_bovw_feature = orb_features(test)[1] 
print("image Feature extractor finish")
def kmeans(k, descriptor_list):
    kmeans = KMeans(n_clusters = k, n_init=10)
    kmeans.fit(descriptor_list)
    visual_words = kmeans.cluster_centers_ 
    return visual_words

print("dictionary words start")
# Takes the central points which is visual words    
visual_words = kmeans(150, descriptor_list) 

print("dictionary words finish")
def find_index(vector1, vector2):

    distanceList = {}

    for ndx, val in enumerate(vector2):
        distanceD = distance.euclidean(vector1, val)
        distanceList[ndx] = distanceD

    # Then find minimum value and its key.
    index = min(distanceList, key=lambda k: distanceList[k])
    return index

def image_class(all_bovw, centers):
    dict_feature = {}
    for key,value in all_bovw.items():
        category = []
        for img in value:
            histogram = np.zeros(len(centers))
            for each_feature in img:
                ind = find_index(each_feature, centers)
                histogram[ind] += 1
            category.append(histogram)
        dict_feature[key] = category
    return dict_feature

print("Histogram start")
# Creates histograms for train data    
bovw_train = image_class(all_bovw_feature, visual_words) 
bovw_test = image_class(test_bovw_feature, visual_words)
print("histogram finish")
# Returns an array that holds number of test images, number of correctly predicted images and records of class based images respectively
# Call the knn function    
#results_bowl = knn(bovw_train, bovw_test) 

def knn(images, tests):
    num_test = 0
    correct_predict = 0
    class_based = {}
    
    for test_key, test_val in tests.items():
        class_based[test_key] = [0, 0] # [correct, all]
        for tst in test_val:
            predict_start = 0
            #print(test_key)
            minimum = 0
            key = "a" #predicted
            for train_key, train_val in images.items():
                for train in train_val:
                    if(predict_start == 0):
                        minimum = distance.euclidean(tst, train)
                        #minimum = L1_dist(tst,train)
                        key = train_key
                        predict_start += 1
                    else:
                        dist = distance.euclidean(tst, train)
                        #dist = L1_dist(tst,train)
                        if(dist < minimum):
                            minimum = dist
                            key = train_key
            
            if(test_key == key):
                correct_predict += 1
                class_based[test_key][0] += 1
            num_test += 1
            class_based[test_key][1] += 1
            #print(minimum)
    return [num_test, correct_predict, class_based]
    
# Call the knn function    
results_bowl = knn(bovw_train, bovw_test) 

def accuracy(results):
    avg_accuracy = (results[1] / results[0]) * 100
    print("Average accuracy: %" + str(avg_accuracy))
    print("\nClass based accuracies: \n")
    for key,value in results[2].items():
        acc = (value[0] / value[1]) * 100
        print(key + " : %" + str(acc))
        
# Calculates the accuracies and write the results to the console.       
accuracy(results_bowl) 


data = []
labels = []
for key in bovw_train:
    for jj in range(len(bovw_train[key])):
        data.append(bovw_train[key][jj])
        labels.append(key)
        


import mahotas
from LocalBinaryPattern import LocalBinaryPatterns
bins = 8
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []
train_labels = os.listdir(train_path)
bins = 8
fixed_size       = tuple((224, 224))
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

# feature-descriptor-3: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

#feature-descriptor-4: Haralick Texture
def fd_haralick(image):
    #convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis = 0)
    return haralick

data = []
labels = []

for training_name in train_labels:
    #join the training data path and each training folder
    dir = os.path.join(train_path, training_name)
    k=0
    current_label = training_name
    #loop over the images in each sub-folder
    for file in os.listdir(dir):
        imageT = cv2.imread(os.path.join(dir, file))
        image = cv2.resize(imageT, fixed_size)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = desc.describe(gray)        
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick = fd_haralick(image)
        fv_histogram = fd_histogram(image)      
        global_feature = np.hstack([bovw_train[current_label][k],hist,fv_haralick,fv_histogram,fv_hu_moments])
        labels.append(current_label)
        data.append(global_feature)
        k=k+1


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold

scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_feature = scaler.fit_transform(data)

test_size = 0.1
scoring = "accuracy"
trainDataGlobal, testDataGlobal,trainLabelsGlobal, testLabelsGlobal = train_test_split(np.array(data),np.array(labels),test_size = test_size,random_state = 42)
model = SVC(random_state = 42)
model.fit(trainDataGlobal, trainLabelsGlobal)
svm_predict= model.predict(testDataGlobal)
acc_svm = model.score(testDataGlobal,testLabelsGlobal)
print("Accuracy on svm: ",acc_svm)

seed=42
num_trees = 10
models = []
models.append(('RF',RandomForestClassifier(n_estimators = num_trees, random_state = seed)))
models.append(('SVM',SVC(random_state = seed)))
models.append(('LR',LogisticRegression(random_state = seed)))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))

results = []
names = []
for name, model in models: 
    kfold =KFold(n_splits = 10, random_state = seed)
    cv_results = cross_val_score(model, trainDataGlobal,trainLabelsGlobal, cv = kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
     