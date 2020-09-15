# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 00:29:47 2020

@author: sami
"""

from LocalBinaryPattern import LocalBinaryPatterns
import cv2
import os
import mahotas
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import MinMaxScaler

def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    

#feature-descriptor-2: Color Histogram
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

train_path = "train"

# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []
train_labels = os.listdir(train_path)
bins = 8
fixed_size       = tuple((224, 224))
# sort the training labels
train_labels.sort()
print(train_labels)

#Extract features from image
for training_name in train_labels:
    #join the training data path and each training folder
    dir = os.path.join(train_path, training_name)
    
    current_label = training_name
    #loop over the images in each sub-folder
    for file in os.listdir(dir):
        # load the image, convert it to grayscale, and describe it
        imageT = cv2.imread(os.path.join(dir, file))
        image = cv2.resize(imageT, fixed_size)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
       #LBP fetaure 
        hist = desc.describe(gray)
        
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick = fd_haralick(image)
        fv_histogram = fd_histogram(image)
        
        global_feature = np.hstack([hist,fv_haralick,fv_histogram,fv_hu_moments])
#        global_feature = np.hstack(hog_image)
        labels.append(current_label)
        data.append(global_feature)

scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_feature = scaler.fit_transform(data)       
# train a Linear SVM on the data
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


test_size = 0.1
scoring = "accuracy"

trainDataGlobal, testDataGlobal,trainLabelsGlobal, testLabelsGlobal = train_test_split(np.array(rescaled_feature),np.array(labels),test_size = test_size,random_state = 42)

model = SVC(C=1,gamma=1e-09,random_state = 42)
#kfold =KFold(n_splits = 200, random_state = 42)
#cv_results = cross_val_score(model, trainDataGlobal,trainLabelsGlobal, cv = kfold, scoring = scoring)
model.fit(trainDataGlobal, trainLabelsGlobal)

svm_predict= model.predict(testDataGlobal)
acc_svm = model.score(testDataGlobal,testLabelsGlobal)
print("Accuracy on svm: ",acc_svm)

#confusion metrix
cm_svm = confusion_matrix(testLabelsGlobal,svm_predict)
plot_confusion_matrix(cm_svm, normalize= False,target_names=train_labels, 
                      title = "Confusion Matrix")


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot
models = []
seed=42
num_trees = 10
models.append(('SVM',SVC(random_state = seed)))
models.append(('LR',LogisticRegression(random_state = seed)))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier(random_state = seed)))
models.append(('RF',RandomForestClassifier(n_estimators = num_trees, random_state = seed)))
models.append(('NB', GaussianNB()))

results = []
names = []
for name, model in models: 
    kfold =KFold(n_splits = 10, random_state = seed)
    cv_results = cross_val_score(model, trainDataGlobal,trainLabelsGlobal, cv = kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
#boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('machine Learning algorithm comparison')
ax = fig.add_subplot(111)    
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

modelRF=RandomForestClassifier(n_estimators = num_trees, random_state = seed)
modelRF.fit(trainDataGlobal, trainLabelsGlobal)
rf_predict= modelRF.predict(testDataGlobal)
acc_rf = modelRF.score(testDataGlobal,testLabelsGlobal)
print("Accuracy on svm: ",acc_rf)

#confusion metrix
cm_rf = confusion_matrix(testLabelsGlobal,rf_predict)
plot_confusion_matrix(cm_rf, normalize= False,target_names=train_labels, 
                      title = "Confusion Matrix")

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

precision = dict()
recall = dict()
average_precision = dict()
y_test=[]
y_score=[]
for i in range(len(testLabelsGlobal)):
    if testLabelsGlobal[i]=='fake':
        y_test.append(0)
    else:
        y_test.append(1)

for i in range(len(rf_predict)):
    if rf_predict[i]=='fake':
        y_score.append(0)
    else:
        y_score.append(1)
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test,y_score)
average_precision["micro"] = average_precision_score(y_test, y_score,average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

plt.step(recall['micro'], precision['micro'], where='post')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    .format(average_precision["micro"]))

from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test, y_score)
print(mse)

from sklearn.metrics import f1_score
f1=f1_score(y_test, y_score)
print(f1)

testPath= 'test\\fake'
dir = os.path.join(testPath)
#start
#prediction_arra=[]
#countR=0 #checking counts from predicted real 
#countRO=0 #give real image count from test set
##
#for i in range(len(testLabelsGlobal)):
#    if str(testLabelsGlobal[i]) == 'real':
#        countRO=countRO+1
#        
#for i in range(len(testDataGlobal)):
#    #test each image from the test dataset
#    prediction = model.predict(testDataGlobal[i].reshape(1, -1)) 
#    if str(prediction[0]) == 'real':
#        countR=countR+1
#    prediction_arra.append(str(prediction[0]))
#print("predicted Real image from data set: ",countR)
#print("Labels Real image from test set: ",countRO)
#end

print("end")
