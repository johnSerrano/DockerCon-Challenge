from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D

def add_layer_pool(model, layer, dataset, lastlayer):
    if not lastlayer:
        model.add(MaxPooling2D(
        pool_size=(layer["pool-y"], layer["pool-x"]),
        input_shape=dataset["input_shape"]))
    else:
        model.add(MaxPooling2D(pool_size=(layer["pool-y"], layer["pool-x"])))


def add_layer_drop(model, layer, dataset, lastlayer):
    model.add(Dropout(layer["dropout-rate"]))


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
            subsample=(layer["stride-y"],layer["stride-x"]),
            input_shape=dataset["input_shape"]))
        model.add(Activation("relu"))
    else:
        model.add(Convolution2D(layer["conv-size"],
            layer["conv-y"],
            layer["conv-x"],
            subsample=(layer["stride-y"],layer["stride-x"])))
        model.add(Activation("relu"))
