# ImageClassification
### Gender Classifier
A Gender Classifier was built using only Convolutional Neural Network. Only two Convolutional Layer was enough for this classifier. This project was built on TFLearn.

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

