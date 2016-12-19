from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import Callback
import numpy as np
import random

from helpers.dataset_helpers import get_dataset
from helpers.image_helpers import png_l, png_rgb
from helpers.layer_helpers import (
    add_layer_conv2d,
    add_layer_drop,
    add_layer_dense,
    add_layer_pool
)

# Number of sample predictions display on the web gui
NUM_PREDICTIONS = 9

def process_network(JSON, callbacks):
    try:
        #TODO check JSON for validity
        room = JSON["room"]
        curr_epoch = 0

        class KerasCallback(Callback):
            def on_epoch_end(self, epoch, logs={}):
                self.values = {
                    "current_epoch": curr_epoch,
                    "total_epochs":  JSON["iterations"],
                    "loss":          logs.get('loss'),
                    "acc":           logs.get('acc'),
                    "val_loss":      logs.get('val_loss'),
                    "val_acc":       logs.get('val_acc'),
                    "done":          False,
                    "predictions":   [],
                }
                for i in range(NUM_PREDICTIONS):
                    d = sample_result(dataset, model, i, room)
                    self.values["predictions"].append(d)
                callbacks["progress"](self.values, room)

        cbk = KerasCallback()
        model, dataset = create_model(JSON, callbacks["loaded"], room)

        for i in range(JSON["iterations"] / dataset["epochs_until_report"]):
            curr_epoch += 1
            model.fit(
                dataset["x_train"],
                dataset["y_train"],
                validation_data=(dataset["x_test"], dataset["y_test"]),
                nb_epoch=dataset["epochs_until_report"],
                batch_size=dataset["batch_size"],
                verbose=0,
                callbacks=[cbk],
                shuffle=True)
            result = model.evaluate(
                dataset["x_test"],
                dataset["y_test"],
                batch_size=dataset["batch_size"],
                verbose=0,
                sample_weight=None
                )

    except Exception as e:
        callbacks["error"]({"error msg": str(e)}, room)
        return

    cbk.values["done"] = True
    callbacks["progress"](cbk.values, room)



def create_model(JSON, callback_loaded, room):
    dataset = get_dataset(JSON)
    model = Sequential()
    callback_loaded(JSON, room)
    last_layer = None

    layers = {
        "conv2d": add_layer_conv2d,
        "dense": add_layer_dense,
        "pool": add_layer_pool,
        "drop": add_layer_drop,
    }

    multi_dimensional_layers = [
        "conv2d",
        "pool",
    ]

    for layer in JSON["layers"]:
        if (last_layer
            and last_layer["type"] in multi_dimensional_layers
            and layer["type"] not in multi_dimensional_layers):
            model.add(Flatten())
        layers[layer["type"]](model, layer, dataset, last_layer)
        last_layer = layer

    if JSON["layers"][-1]["type"] in ["conv2d", "pool"]:
        model.add(Flatten())

    model.add(Dense(dataset["nb_classes"]))
    model.add(Activation("softmax"))

    model.compile(loss=JSON["loss"],
                  optimizer='adadelta',
                  metrics=['accuracy'])

    # TODO: fix callback system
    # redundant call after slow operation to increase chance of results page
    # having loaded by now. better system necessary, perhaps send on room join
    callback_loaded(JSON, room)
    return model, dataset


def sample_result(dataset, model, count, room):
    random_selection = random.randrange(dataset["x_test"].shape[0])

    image_location = str(room) + "_" + str(count)

    #TODO: support alpha channel
    #TODO: don't read from file. wtf is that about
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
