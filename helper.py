## Libraries necessary for defining the model in keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model

## Other Libraries
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os

###############
#### MODEL ####
###############

def mlp_network(dim, regress=False):
	model = Sequential()
	model.add(Dense(8, input_dim=dim, activation="relu"))
	model.add(Dense(4, activation="relu"))

	## Verify the regression node
	if regress:
		model.add(Dense(1, activation="linear"))

	return model

def conv_network(width, height, depth, filters=(16, 32, 64), regress=False):
	## Channels-last based on Tensorflow
	input_shape = (height, width, depth)
	chanDim = -1

	## Model input
	inputs = Input(shape=input_shape)

	## Loop over filters
	for (i, f) in enumerate(filters):
		if i == 0:
			x = inputs

		## CONV => RELU => BN => POOL
		x = Conv2D(f, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

	## Flatten the volume, then FC => RELU => BN => DROPOUT
	x = Flatten()(x)
	x = Dense(16)(x)
	x = Activation("relu")(x)
	x = BatchNormalization(axis=chanDim)(x)
	x = Dropout(0.5)(x)

	## Another FC layer
	x = Dense(4)(x)
	x = Activation("relu")(x)

    ## Verify the regression node
	if regress:
		x = Dense(1, activation="linear")(x)

	model = Model(inputs, x)
	return model

#####################
#### DATA LOADER ####
#####################

def house_details(inputPath):
	columns = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
	df = pd.read_csv(inputPath, sep=" ", header=None, names=columns)

	zipcodes = df["zipcode"].value_counts().keys().tolist()
	counts = df["zipcode"].value_counts().tolist()

	for (zipcode, count) in zip(zipcodes, counts):
		if count < 25:
			idxs = df[df["zipcode"] == zipcode].index
			df.drop(idxs, inplace=True)

	return df

def process_house_details(df, train, test):
	continuous = ["bedrooms", "bathrooms", "area"]

	## Min-Max scaling on continuous feature columns
	cs = MinMaxScaler()
	train_cont = cs.fit_transform(train[continuous])
	test_cont = cs.transform(test[continuous])

	zip_binarizer = LabelBinarizer().fit(df["zipcode"])
	train_cat = zip_binarizer.transform(train["zipcode"])
	test_cat = zip_binarizer.transform(test["zipcode"])

	## Training and Testing data
	X_train = np.hstack([train_cat, train_cont])
	X_test = np.hstack([test_cat, test_cont])

	return (X_train, X_test)

def load_images(df, inputPath):
	images = []
	for i in df.index.values:
		## Images for the house and sort the file paths and ensuring the order
		Path_base = os.path.sep.join([inputPath, "{}_*".format(i + 1)])
		Path_house = sorted(list(glob.glob(Path_base)))

		## Define output image
		input_Images = []
		output_Image = np.zeros((64, 64, 3), dtype="uint8")

		for hp in Path_house:
			## load the input image, resize it to be 32 32
			image = cv2.imread(hp)
			image = cv2.resize(image, (32, 32))
			input_Images.append(image)

		output_Image[0:32, 0:32] = input_Images[0]
		output_Image[0:32, 32:64] = input_Images[1]
		output_Image[32:64, 32:64] = input_Images[2]
		output_Image[32:64, 0:32] = input_Images[3]

		images.append(output_Image)

	return np.array(images)
