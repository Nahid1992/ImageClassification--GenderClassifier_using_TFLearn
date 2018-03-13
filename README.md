# Image Classification
### Gender Classifier
A python implemntation of Gender Classification using Convolutional Neural Network. Only two Convolutional Layer was enough for this classifier. This project was built on TFLearn. First tried training with lots of hidden layers (exp: alexnet or own-architech). The performance was not so much impressive. But applying only two conv layer, the performance started to grow rapidly. Data Augmentation was not needed as the number of samples chosen from the dataset was enough.   

### Database
The gender imageset was downloaded from [CACD](http://bcsiriuschen.github.io/CARC/) dataset. 

	Each Image Size = 128 X 128
	Training Set = 28,000 images
	Validation Set = 12,000 images
	
### Data Preparation
Each CACD Face images were prepared by running [Viola Jones Face detector](https://www.mathworks.com/help/vision/ref/vision.cascadeobjectdetector-system-object.html).
Implementation of Viola Jones is not provided in this repository. This was done earlier in one of other my project using MatLab. Luckily I stored the training, validation and testing set. 

### Dependencies
* Python 3.6.2
* Numpy
* TensorFlow
* TFLearn
* MatplotLib
* SciPy

