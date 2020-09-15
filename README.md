# FaceSpoofingDetection

Step by Step Running of the Project:

1.Download the data source:
links: https://www.kaggle.com/ciplab/real-and-fake-face-detection

Other sources: 

links: https://www.kaggle.com/c/deepfake-detection-challenge/data
https://www.kaggle.com/xhlulu/flickrfaceshq-dataset-nvidia-part-1



Note:- put the downloaded image as below folder strcuture

-train

	-fake
	
	-real


2.Project Running

Requirement to run:
	python 3.7.4
	
	scikit-learn 0.21.3
	
	scikit-image 0.16.2
	
	mahotas 1.4.9
	
	matplotlib 3.1.1
	
	opencv-contrib-python 4.2.0.32
	
	opencv-python 4.2.0.32
	
	scipy 1.3.1


(I)preprocessing steps which might not need to run for you
	a) Ensure OpenCV installed in your python environemnt 
	b) create a directory structure like 
		-train_cropped
			-fake
			-real
	c) Run the backgroundconvert.py file
	d) remove the main train directory and rename the train_cropped as train.
(II)imageSegmentation.py
	a) will generate the lBP histogram diagram and the process of texture feature extraction steps
	b) used for describing methodology in report

(III) Ensure LocalBinaryPattern.py and imgaeClassificationUsingLBP.py file in same directory
	a)install mahotas in mpython environment [command(pip install mahotas)]
	b)install sklearn, skimage 
	c) run the imgaeClassificationUsingLBP.py (it will take time to finish)
	d) after running, the result would be generated.
(IV) orbImageClassifier.py (it took almost 36 hours of waiting but didn't finish)
	a)exclude the feature after preprocessing.
