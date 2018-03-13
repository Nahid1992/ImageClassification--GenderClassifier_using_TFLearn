# ImageClassification
### Gender Classifier
A Gender Classifier was built using only Convolutional Neural Network. Only two Convolutional Layer was enough for this classifier. This project was built on TFLearn. First tried training with lots of hidden layers (exp: alexnet or own-architech). The performance was not so much impressive. But trying with only two conv layer the performance started to grow rapidly. Data Augmentation was not need as number of samples chosen from the dataset was good enough.   

### Database
The gender imageset was downloaded from [CACD](http://bcsiriuschen.github.io/CARC/) dataset. 

	Each Image Size = 128 X 128
	Training Set = 28,000 images
	Testing Set = 12,000 images
	
### Data Preparation
Each CACD Face images were prepared by running [Viola Jones Face detector](https://www.mathworks.com/help/vision/ref/vision.cascadeobjectdetector-system-object.html).
Implementation of Viola Jones is not provided in this repository. This was done earlier one of my project using MatLab. 

### Dependencies
* Python 3.6.2
* Numpy
* TensorFlow
* TFLearn
* MatplotLib
* SciPy

