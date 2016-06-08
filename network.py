from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import Callback
import numpy as np

def process_network(JSON, callback_sockets, callback_loaded):
	curr_epoch = 0

	# process data
	dataset = None
	result = []
	model = Sequential()
	room = JSON["room"]

	if JSON["dataset"] == "mnist":
		dataset = dataset_mnist()
	elif JSON["dataset"] == "cifar 10":
		dataset = dataset_cifar10()

	if not dataset:
		#should send error data to socket callback maybe
		return

	callback_loaded(JSON, room)

	class Loss(Callback):
		def on_epoch_end(self, epoch, logs={}):
			self.values = {"current_epoch": curr_epoch, "total_epochs": JSON["iterations"], "loss": logs.get('loss'), "acc": logs.get('acc'), "val_loss": logs.get('val_loss'), "val_acc": logs.get('val_acc'), "done": False,}
			callback_sockets(self.values, room)

	# generate layers, only dense for now
	for i, layer in enumerate(JSON["layers"]):
		if i == 0:
			model.add(Dense(output_dim=layer["size"], input_shape=dataset["input_shape"]))
		else:
			model.add(Dense(output_dim=layer["size"], input_shape=(JSON["layers"][i-1]["size"],)))
		model.add(Activation("relu"))
	model.add(Dense(dataset["nb_classes"]))
	model.add(Activation("softmax"))

	# use adadelta and cat_xent for now
	model.compile(loss='mean_absolute_error',
				  optimizer='adadelta',
				  metrics=['accuracy'])

	# print model.summary()

	cbk = Loss()
	# run callback with socket results
	for i in range(JSON["iterations"] / dataset["epochs_until_report"]):
		curr_epoch += 1
		model.fit(dataset["x_train"],
				  dataset["y_train"],
				  validation_data=(dataset["x_test"], dataset["y_test"]),
				  nb_epoch=dataset["epochs_until_report"],
				  batch_size=dataset["batch_size"],
				  verbose=0,
				  callbacks=[cbk],
				  shuffle=True)
		result = model.evaluate(dataset["x_test"],
								dataset["y_test"],
								batch_size=dataset["batch_size"],
								verbose=0,
								sample_weight=None)
	cbk.values["done"] = True
	callback_sockets(cbk.values, room)

	return 'Test accuracy: ' + str(result[1]) + '\n'


def dataset_mnist():
	from keras.datasets import mnist
	dataset = {
		"name": "mnist"
	}
	img_rows, img_cols = 28, 28
	dataset["nb_classes"] = 10
	dataset["batch_size"] = 128
	dataset["epochs_until_report"] = 1
	dataset["input_shape"] = (28*28,)

	(dataset["x_train"], dataset["y_train"]), (dataset["x_test"], dataset["y_test"]) = mnist.load_data()
	dataset["x_train"] = dataset["x_train"].reshape(dataset["x_train"].shape[0], img_rows * img_cols)
	dataset["x_test"] = dataset["x_test"].reshape(dataset["x_test"].shape[0], img_rows * img_cols)
	dataset["x_train"] = dataset["x_train"].astype('float32')
	dataset["x_test"] = dataset["x_test"].astype('float32')
	dataset["x_train"] /= 255
	dataset["x_test"] /= 255

	dataset["y_train"] = np_utils.to_categorical(dataset["y_train"], dataset["nb_classes"])
	dataset["y_test"] = np_utils.to_categorical(dataset["y_test"], dataset["nb_classes"])

	return dataset

def dataset_cifar10():
	from keras.datasets import cifar10
	dataset = {
		"name": "CIFAR 10"
	}
	img_rows, img_cols = 32, 32
	img_channels = 3
	dataset["nb_classes"] = 10
	dataset["batch_size"] = 128
	dataset["epochs_until_report"] = 1
	dataset["input_shape"] = (img_rows * img_cols * img_channels,)

	(dataset["x_train"], dataset["y_train"]), (dataset["x_test"], dataset["y_test"]) = cifar10.load_data()
	dataset["x_train"] = dataset["x_train"].reshape(dataset["x_train"].shape[0], img_rows * img_cols * img_channels)
	dataset["x_test"] = dataset["x_test"].reshape(dataset["x_test"].shape[0], img_rows * img_cols * img_channels)
	dataset["x_train"] = dataset["x_train"].astype('float32')
	dataset["x_test"] = dataset["x_test"].astype('float32')
	dataset["x_train"] /= 255
	dataset["x_test"] /= 255

	dataset["y_train"] = np_utils.to_categorical(dataset["y_train"], dataset["nb_classes"])
	dataset["y_test"] = np_utils.to_categorical(dataset["y_test"], dataset["nb_classes"])

	return dataset
