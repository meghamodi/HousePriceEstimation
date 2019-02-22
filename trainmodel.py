from helper import house_details, process_house_details, load_images
from helper import mlp_network, conv_network
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate
import numpy as np
import locale
import os

## Dataset Location
housedata = 'HousePriceCalculation/housedata/'

## Read the txt file tto get information on each house
df = house_details(housedata + 'HousesInfo.txt')

## Load the house images
images = load_images(df, housedata)
## Bring to 0 to 1 range
images = images / 255.0

## Train Test Split
split = train_test_split(df, images, test_size=0.25, random_state=42)
(train_attr_X, test_attr_X, train_images_X, test_images_X) = split

## Scaling the target variable (house price)
max_price = train_attr_X["price"].max()
Y_train = train_attr_X["price"] / max_price
Y_test = test_attr_X["price"] / max_price

## Processing the data, one-hot encoding for categorical features
(train_attr_X, test_attr_X) = process_house_details(df,
	train_attr_X, test_attr_X)

## Define the model
mlp = mlp_network(train_attr_X.shape[1], regress=False)
cnn = conv_network(64, 64, 3, regress=False)

## Taking the combined output
combined_Input = concatenate([mlp.output, cnn.output])

##The final dense layer is regression head
x = Dense(4, activation="relu")(combined_Input)
x = Dense(1, activation="linear")(x)

## Categorical/numerical data on the MLP input and Images on the CNN input, Final output is single value (price of the house)
model = Model(inputs=[mlp.input, cnn.input], outputs=x)

## Model architecture
plot_model(model, to_file='HousePriceCalculation/houseKerasmodel.png', show_shapes=True, show_layer_names=True)

## Mean absolute percentage error as the loss criteria
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

## Training
print("Start Training...")
model.fit(
	[train_attr_X, train_images_X], Y_train,
	validation_data=([test_attr_X, test_images_X], Y_test),
	epochs=200, batch_size=8)

## Prediction
print("Predicting House Prices...")
preds = model.predict([test_attr_X, test_images_X])


## Absolute percentage diference
dif = preds.flatten() - Y_test
percent_dif = (dif / Y_test) * 100
abs_percent_dif = np.abs(percent_dif)

# Calculate Mean and Standard Deviation
mean = np.mean(abs_percent_dif)
std = np.std(abs_percent_dif)

## Statistics for the model
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("Avg. house price: {}, std house price: {}".format(
	locale.currency(df["price"].mean(), grouping=True),
	locale.currency(df["price"].std(), grouping=True)))
print("mean: {:.2f}%, std: {:.2f}%".format(mean, std))
