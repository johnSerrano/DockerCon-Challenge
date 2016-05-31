from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import Callback
import numpy as np

def process_network(JSON, callback_sockets):
	curr_epoch = 0

	# process data
	dataset = None
	nb_classes = None
	batch_size = None
	epochs_until_report = None
	result = []
	input_dim = (None, None)
	model = Sequential()

	if JSON["dataset"] == "mnist":
		from keras.datasets import mnist
		img_rows, img_cols = 28, 28
		nb_classes = 10
		batch_size = 128
		epochs_until_report = 1
		input_dim=28*28

		(X_train, y_train), (X_test, y_test) = mnist.load_data()
		X_train = X_train.reshape(X_train.shape[0], img_rows * img_cols)
		X_test = X_test.reshape(X_test.shape[0], img_rows * img_cols)
		X_train = X_train.astype('float32')
		X_test = X_test.astype('float32')
		X_train /= 255
		X_test /= 255

		y_train = np_utils.to_categorical(y_train, nb_classes)
		y_test = np_utils.to_categorical(y_test, nb_classes)

		dataset = {
			"name": "mnist",
			"x_train": X_train,
			"y_train": y_train,
			"x_test": X_test,
			"y_test": y_test,
		}

	if not dataset:
		#should send error data to socket callback maybe
		return

	class Loss(Callback):
		def on_epoch_end(self, epoch, logs={}):
			self.values = {"current_epoch": curr_epoch, "total_epochs": JSON["iterations"], "loss": logs.get('loss'), "acc": logs.get('acc'), "val_loss": logs.get('val_loss'), "val_acc": logs.get('val_acc'), "done": False,}
			callback_sockets(self.values)

	# generate layers, only dense for now
	for i, layer in enumerate(JSON["layers"]):
		if i == 0:
			model.add(Dense(output_dim=layer["size"], input_dim=input_dim))
		else:
			model.add(Dense(output_dim=layer["size"], input_dim=JSON["layers"][i-1]["size"]))
		model.add(Activation("relu"))
	model.add(Dense(nb_classes))
	model.add(Activation("softmax"))

	# use adadelta and cat_xent for now
	model.compile(loss='mean_absolute_error',
				  optimizer='adadelta',
				  metrics=['accuracy'])

	print model.summary()

	cbk = Loss()
	# run callback with socket results
	for i in range(JSON["iterations"] / epochs_until_report):
		curr_epoch += 1
		model.fit(dataset["x_train"],
				  dataset["y_train"],
				  validation_data=(dataset["x_test"], dataset["y_test"]),
				  nb_epoch=epochs_until_report,
				  batch_size=batch_size,
				  verbose=0,
				  callbacks=[cbk],
				  shuffle=True)
		result = model.evaluate(dataset["x_test"],
								dataset["y_test"],
								batch_size=batch_size,
								verbose=0,
								sample_weight=None)
	cbk.values["done"] = True
	callback_sockets(cbk.values)

	return 'Test accuracy: ' + str(result[1]) + '\n'
