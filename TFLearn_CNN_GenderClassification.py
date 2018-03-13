from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt
import numpy as np
import h5py
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_utils import build_hdf5_image_dataset
from tflearn.layers.normalization import local_response_normalization
import scipy.io as sio
import cv2

#Restarting karnel
import tensorflow as tf
print('Kernel Restarting..')
tf.reset_default_graph()
print('Kernel Restarted..')

img_height = 128
img_width = 128
nb_classes = 2

def convertToOneHot(vector, num_classes=None):
    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)

def data_from_mat(filename):
    mat_content = sio.loadmat(filename)
    GENDER = mat_content['GENDER']
    GENDER = GENDER[0,0]
    tempX = GENDER["I"]
    tempY = GENDER["labels"]

    imgList = []
    imgLabel = []
    for i in range(0,tempX.shape[1]):
    	imgList.append(tempX[0,i])
    	label = tempY[i]
    	label = label[0]
    	imgLabel.append(label)
    	#print('Image_Stored ' + str(i))

    X = np.array(imgList)
    Y = np.array(imgLabel)
    X,Y = tflearn.data_utils.shuffle(X,Y)

    valX = X[-12000:]
    valY = Y[-12000:]
    trainX = X[:28000]
    trainY = Y[:28000]

    trainX = trainX.reshape(-1,img_width,img_height,1)
    valX = valX.reshape(-1,img_width,img_height,1)
    trainY = convertToOneHot(trainY,nb_classes)
    valY = convertToOneHot(valY,nb_classes)

    trainX = trainX/255
    valX = valX/255

    print('Training + Validation Data ready to Train')
    return trainX,trainY,valX,valY

def Data_Generate(dataset_file,type,task):
    output_pathX = 'data/'+type+'_dataX.h5'
    output_pathY = 'data/'+type+'_dataY.h5'
    if task == 'save':
    	train_file = open(dataset_file,'r')

    	#X
    	imgList = []
    	for index in train_file:
    		imgFile = index[0:-2]
    		if imgFile[-1] == 'a':
    			continue
    		original_img = plt.imread(imgFile)
    		imgList.append(original_img)
    		print('Stored -> ' + imgFile)
    	train_file.close()
    	#Store imglist into h5
    	h5f = h5py.File(output_pathX, 'w')
    	h5f.create_dataset('X', data=imgList)
    	h5f.close()

    	#Y
    	labelList = []
    	for index in train_file:
    		if index[-1] == 'n':
    			continue
    		labelList.append(index[-1])
    	#store labellist into h5
    	h5f = h5py.File(output_pathY, 'w')
    	h5f.create_dataset('Y', data=labelList)
    	h5f.close()


    if task == 'load':
    	#X
    	h5f = h5py.File(output_pathX, 'r')
    	X = h5f['X']
    	#Y
    	h5f = h5py.File(output_pathY,'r')
    	Y = h5f['Y']

    #shuffle
    X,Y = tflearn.data_utils.shuffle(X,Y)
    X = X/255
    return X,Y

def create_model_AlexNet():
    # Building 'AlexNet'
    '''
    # Real-time data preprocessing
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # Real-time data augmentation
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)
    '''

    network = input_data(shape=[None, img_width, img_height,1])

    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)

    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)

    network = fully_connected(network, 2, activation='softmax')

    model = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

    return model

#def create_model_InceptionV3():


def create_own_model():

    # Real-time data preprocessing
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # Real-time data augmentation
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)


    # Convolutional network building
    network = input_data(shape=[None, img_width, img_height,1],data_preprocessing=img_prep,data_augmentation=img_aug)

    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 128, 3, activation='relu')
    network = conv_2d(network, 128, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, nb_classes, activation='softmax')

    model = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return model

def create_own_model_two_conv2():
    network = input_data(shape=[None, img_width, img_height,1])

    network = conv_2d(network, 64, 3, strides=2, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = conv_2d(network, 64, 3, strides=2, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = fully_connected(network, 128, activation='relu')
    network = fully_connected(network, nb_classes, activation='softmax')
    model = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)

    return model


def train_model(model,x_train,y_train,x_val,y_val):
	model = tflearn.DNN(model, tensorboard_verbose=3)

	model.fit(x_train, y_train, n_epoch=5, validation_set=(x_val, y_val), shuffle=False,
			show_metric=True, batch_size=50, snapshot_step=25,
			snapshot_epoch=False, run_id='Gender_tflearn_run01')

	#save Model
	model.save('models/tflearn_genderClassification_model.model')

def load_model():
	model.load('models/tflearn_genderClassification_model.model')

def main():

    filename = 'data/GENDER.mat'
    x_train, y_train, x_val, y_val = data_from_mat(filename)
    print('x_train = ', str(x_train.shape))
    print('y_train = ', str(y_train.shape))
    print('x_val = ', str(x_val.shape))
    print('y_val = ', str(y_val.shape))
    print('Data is Ready...')

    #model = create_model()
    #model = create_model_AlexNet()
    #model = create_model_InceptionV3()
    model = create_own_model_two_conv2() # WORKING
    print('Model Created...')

    print('')
    print('Training Started...')
    train_model(model,x_train,y_train,x_val,y_val)
    print('Training Done & Model Saved...')
    print('')

    breakPOINT = 1

if __name__== "__main__":
  main()
