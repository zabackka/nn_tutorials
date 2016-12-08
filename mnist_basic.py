import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

# seed the random number generator with the same seed every time, 
# so we can duplicate results (reproducibility)
seed = 7
numpy.random.seed(seed)

###############
## load data ##
###############

# load the training and test data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# calculate the number of pixels per image (# of input nodes required)
# X_train is a 3-d array (instance * image width * image height)
# X_train.shape[0] gives the # of instances
# X_train.shape[1] gives the width of the image
# X_train.shape[2] gives the height of the image
n_pixels = X_train.shape[1] * X_train.shape[2]

# reshape X_train/test so that it is a 2-d array (60000 * 784)
# 60000: # of instances in the set
# 784: 28*28 (the total # of pixels in a single image)
# each row now contains a vector that holds all the pixels for a single image
# NOTE: we change the pixel values from int to float (for normalization)
X_train = X_train.reshape(X_train.shape[0], n_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], n_pixels).astype('float32')

# transform categorical output data using "one-hot encoding"
# transforms the single digit classification into a vector
# ex. 5 -> [0 0 0 0 0 1 0 0 0 0]
# class //	0 1 2 3 4 5 6 7 8 9
# each slot of the vector represents a classification category
# the 1 represents the class that particular examples belongs to
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# normalize input data (from 0-255 to 0-1)
X_train = X_train / 255
X_test = X_test / 255

# the number of classes is equal to the shape[1] of y_train/test
n_classes = y_test.shape[1]




#####################
## structure model ##
#####################

def basic_nn():
	model = Sequential()
	
	# create the hidden layer
	# number of nodes = # of input nodes (pixels in the image)
	# rectifier activation function is used (will return a non-negative number)
	model.add(Dense(n_pixels, input_dim=n_pixels, init='normal', activation='relu'))
	
	# create the output layer
	# number of nodes = 10 (one for each classification)
	# softmax activation function (return a number between 0 & 1)
	# the number outputted for each node represents the net's predicted probablity
	#	that the image is of that node class
	model.add(Dense(n_classes, init='normal', activation='softmax'))

	# compile the model
	# loss function = logarithmic loss
	# optimizing function = ADAM gradient descent
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	return model

#################
## build model ##
#################

# build the model
model = basic_nn()

# fit the model to the MNIST data
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=1, batch_size=200, verbose=2)

# evaluate the model
error = model.evaluate(X_test, y_test, verbose=0)

print("error: %.2f%%" % (100-error[1]*100))



