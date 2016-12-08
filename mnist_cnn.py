
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# seed the random number generator with the same seed every time, 
# so we can duplicate results (allows for reproducibility)
seed = 7
numpy.random.seed(seed)

###############
## load data ##
###############

# load the training and test data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# we want our input dims to be: [channels] [image width] [image height]
# since we are not using RGB (which would make channels=3), our dims are:
# 1 * 28 * 28
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalize input data (from 0-255 to 0-1)
X_train = X_train / 255
X_test = X_test / 255

# transform categorical output data using "one-hot encoding"
# transforms the single digit classification into a vector
# ex. 5 -> [0 0 0 0 0 1 0 0 0 0]
# class //	0 1 2 3 4 5 6 7 8 9
# each slot of the vector represents a classification category
# the 1 represents the class that particular examples belongs to
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# the number of classes is equal to the shape[1] of y_train/test
num_classes = y_test.shape[1]


#####################
## structure model ##
#####################

def cnn_model():
	model = Sequential()
	
	# build convolution layer
	# 32: # of feature maps
	# 5, 5: each feature map is a 5px*5px receptive field
	# border mode -> valid: only compute covolution where the input & filter fully overlap
	# [ i.e. do not perfom zero-padding ]
	# input shape: [channels (1)] [image width (28px)] [image height (28px)]
	# relu: rectifier activation function is used (will return a non-negative number)
	model.add(Convolution2D(32, 5, 5, input_shape=(1, 28, 28), activation='relu'))

	# build max-pool layer
	# compute max-pooling for each 2x2 area
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# build regularization/dropout layer
	# randomly drop 20% of the max-pool layer to prevent overfitting & keep 
	model.add(Dropout(0.2))

	# build flattening layer
	# converted 2D array to a vector, so we can feed it into the fully-connected NN layer
	model.add(Flatten())

	# build a fully-connected layer
	# layer contains 128 neurons & uses the rectifier activation function
	# [ relu returns a non-negative value ]
	model.add(Dense(128, activation='relu'))

	# create the output layer
	# number of nodes = 10 (one for each classification)
	# softmax activation function (return a number between 0 & 1)
	# the number outputted for each node represents the net's predicted probablity
	#	that the image is of that node class
	model.add(Dense(num_classes, activation='softmax'))

	# compile the model
	# loss function = logarithmic loss
	# optimizing function = ADAM gradient descent
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	return model

#################
## build model ##
#################

# build the model
model = cnn_model()

# fit the model to the MNIST data
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)

# evaluate the model
error = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-error[1]*100))



