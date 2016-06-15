from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import Callback
import numpy as np
import png
import random

def process_network(JSON, callback_sockets, callback_loaded):

    #TODO check JSON for validity
    room = JSON["room"]
    curr_epoch = 0

    class Cbk(Callback):
        def on_epoch_end(self, epoch, logs={}):
            self.values = {"current_epoch": curr_epoch,
            "total_epochs": JSON["iterations"],
            "loss": logs.get('loss'),
            "acc": logs.get('acc'),
            "val_loss": logs.get('val_loss'),
            "val_acc": logs.get('val_acc'),
            "done": False,
            "predictions": [],
            }
            for i in range(9):
                d = sample_result(dataset, model, i, room)
                self.values["predictions"].append(d)
            callback_sockets(self.values, room)

    #create model
    model, dataset = create_model(JSON, callback_loaded, room)
    cbk = Cbk()
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


def sample_result(dataset, model, count, room):
    random_selection = random.randrange(dataset["x_test"].shape[0])

    image_location = str(room) + "_" + str(count)

    #TODO support alpha channel
    if dataset["PNG_mode"] == "L":
        png_l(np.copy(dataset["x_test"][random_selection]), dataset, "results/" + image_location)
        data_uri = open("results/" + image_location, 'rb').read().encode('base64').replace('\n', '')
        img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
    elif dataset["PNG_mode"] == "RGB":
        png_rgb(np.copy(dataset["x_test"][random_selection]), dataset, "results/" + image_location)
        data_uri = open("results/" + image_location, 'rb').read().encode('base64').replace('\n', '')
        img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
    else:
        #TODO: return empty image
        return {}

    # TODO: predict only one element
    predictions = model.predict_classes(dataset["x_test"], batch_size=128, verbose=0)
    predict = predictions.tolist()[random_selection]

    return {
        "expected": dataset["classes"][dataset["y_test"].tolist()[random_selection].index(1)],
        "predicted": dataset["classes"][predict],
        "img_tag": img_tag,
    }


def create_model(JSON, callback_loaded, room):
    dataset = get_dataset(JSON)
    model = Sequential()
    callback_loaded(JSON, room)

    layers = {
        "conv2d": add_layer_conv2d,
        "dense": add_layer_dense,
        "pool": add_layer_pool,
    }

    #TODO add layers
    for i, layer in enumerate(JSON["layers"]):
        if i == 0:
            layers[layer["type"]](model, layer, dataset, None)
        else:
            if JSON["layers"][i-1]["type"] in ["conv2d", "pool"] and layer["type"] not in ["conv2d", "pool"]:
                model.add(Flatten())
            layers[layer["type"]](model, layer, dataset, JSON["layers"][i-1])

    model.add(Dense(dataset["nb_classes"]))
    model.add(Activation("softmax"))

    model.compile(loss='mean_absolute_error',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    return model, dataset


def add_layer_pool(model, layer, dataset, lastlayer):
    pass


def add_layer_dense(model, layer, dataset, lastlayer):
    if not lastlayer:
        dataset["x_train"] = dataset["x_train"].reshape(dataset["x_train"].shape[0], dataset["img_rows"] * dataset["img_cols"] * dataset["img_channels"])
        dataset["x_test"] = dataset["x_test"].reshape(dataset["x_test"].shape[0], dataset["img_rows"] * dataset["img_cols"] * dataset["img_channels"])
        model.add(Dense(output_dim=layer["dense-size"], input_shape=(dataset["img_rows"] * dataset["img_cols"] * dataset["img_channels"],)))
        model.add(Activation("relu"))
    else:
        model.add(Dense(output_dim=layer["dense-size"]))
        model.add(Activation("relu"))

#note to self: first layer reshape
def add_layer_conv2d(model, layer, dataset, lastlayer):
    if not lastlayer:
        model.add(Convolution2D(layer["conv-size"],
        layer["conv-y"],
        layer["conv-x"],
        subsample=(layer["stride-y"],
        layer["stride-x"]),
        input_shape=dataset["input_shape"]))
        model.add(Activation("relu"))
    else:
        model.add(Convolution2D(layer["conv-size"],
        layer["conv-y"],
        layer["conv-x"],
        subsample=(layer["stride-y"],layer["stride-x"])))
        model.add(Activation("relu"))


def get_dataset(JSON):
    datasets = {
        "mnist": dataset_mnist,
        "cifar 10": dataset_cifar10,
    }
    if JSON["dataset"] in datasets.keys():
        return datasets[JSON["dataset"]]()
    else:
        raise Exception("Invalid dataset name")


def dataset_mnist():
    from keras.datasets import mnist
    dataset = {
        "name": "mnist"
    }
    dataset["classes"] = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5",
                          6: "6", 7: "7", 8: "8", 9: "9"}
    dataset["img_rows"], dataset["img_cols"] = 28, 28
    dataset["nb_classes"] = 10
    dataset["batch_size"] = 128
    dataset["epochs_until_report"] = 1
    dataset["img_channels"] = 1
    dataset["input_shape"] = (dataset["img_channels"], dataset["img_rows"], dataset["img_cols"])
    dataset["PNG_mode"] = "L"

    (dataset["x_train"], dataset["y_train"]), (dataset["x_test"], dataset["y_test"]) = mnist.load_data()
    dataset["x_train"] = dataset["x_train"].reshape(dataset["x_train"].shape[0], dataset["img_channels"], dataset["img_rows"], dataset["img_cols"])
    dataset["x_test"] = dataset["x_test"].reshape(dataset["x_test"].shape[0], dataset["img_channels"], dataset["img_rows"], dataset["img_cols"])
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
    dataset["classes"] = {0: "Plane", 1: "Car", 2: "Bird", 3: "Cat", 4: "Deer",
                         5: "Dog", 6: "Frog", 7: "Horse", 8: "Ship", 9: "Truck"}
    dataset["img_rows"], dataset["img_cols"] = 32, 32
    dataset["img_channels"] = 3
    dataset["nb_classes"] = 10
    dataset["batch_size"] = 128
    dataset["epochs_until_report"] = 1
    dataset["input_shape"] = (dataset["img_channels"], dataset["img_rows"], dataset["img_cols"])
    dataset["PNG_mode"] = "RGB"

    (dataset["x_train"], dataset["y_train"]), (dataset["x_test"], dataset["y_test"]) = cifar10.load_data()
    dataset["x_train"] = dataset["x_train"].astype('float32')
    dataset["x_test"] = dataset["x_test"].astype('float32')
    dataset["x_train"] /= 255
    dataset["x_test"] /= 255

    dataset["y_train"] = np_utils.to_categorical(dataset["y_train"], dataset["nb_classes"])
    dataset["y_test"] = np_utils.to_categorical(dataset["y_test"], dataset["nb_classes"])

    return dataset

def png_l(arr, dataset, location):
	#grayscale png image
	with open(location, 'wb') as f:
		arr *= 255
		arr = arr.reshape(dataset["img_rows"], dataset["img_cols"])
		w = png.Writer(dataset["img_rows"], dataset["img_cols"], greyscale=True)
		w.write(f, arr)

def png_rgb(arr, dataset, location):
	arr *= 255
	arr = arr.reshape(dataset["img_rows"]*dataset["img_cols"]*dataset["img_channels"])
	pixels = []
	image = []

	for i in range(dataset["img_rows"]*dataset["img_cols"]):
		pixel = [arr[i], arr[i+(dataset["img_rows"]*dataset["img_cols"])],
				 arr[i+(dataset["img_rows"]*dataset["img_cols"]*2)]]
		pixels += [pixel]

	for i in range(dataset["img_rows"]):
		row = []
		for j in range(dataset["img_cols"]):
			row += pixels[(dataset["img_cols"]*i)+j]
		image += [row]

	with open(location, 'wb') as f:
		w = png.Writer(dataset["img_rows"], dataset["img_cols"])
		w.write(f, image)
