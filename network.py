from copy import deepcopy
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import Callback
from threading import Lock
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

# Number of sample predictions displayed on the web gui
NUM_PREDICTIONS = 9

class Network():
    def __init__(self, network):
        #TODO: check network for validity
        self.state = {
            "network": network,
            "current_epoch": 0,
            "total_epochs":  0,
            "loss":          0,
            "acc":           0,
            "val_loss":      0,
            "val_acc":       0,
            "done":          False,
            "predictions":   [],
            "loaded":        False,
        }
        self.dataset = None
        self.state_mutex = Lock()

    def get_state(self):
        self.state_mutex.acquire()
        try:
            return deepcopy(self.state)
        finally:
            self.state_mutex.release()

    def process_network(self):
        class KerasCallback(Callback):
            def on_epoch_end(cbk, epoch, logs={}):
                self.state_mutex.acquire()
                try:
                    self.state["current_epoch"] = curr_epoch
                    self.state["total_epochs"] = self.state["network"]["iterations"]
                    self.state["loss"] = logs.get('loss')
                    self.state["acc"] = logs.get('acc')
                    self.state["val_loss"] = logs.get('val_loss')
                    self.state["val_acc"] = logs.get('val_acc')
                    self.state["done"] = False
                    self.state["error"] = None
                    self.state["loaded"] = True
                    self.state["predictions"] = self.sample_results(model)
                finally:
                    self.state_mutex.release()
        try:
            curr_epoch = 0
            cbk = KerasCallback()
            self.dataset = get_dataset(self.state["network"])
            model = self.create_model()

            for i in range(self.state["network"]["iterations"] / self.dataset["epochs_until_report"]):
                curr_epoch += 1
                model.fit(
                    self.dataset["x_train"],
                    self.dataset["y_train"],
                    validation_data=(self.dataset["x_test"], self.dataset["y_test"]),
                    nb_epoch=self.dataset["epochs_until_report"],
                    batch_size=self.dataset["batch_size"],
                    verbose=0,
                    callbacks=[cbk],
                    shuffle=True)
                result = model.evaluate(
                    self.dataset["x_test"],
                    self.dataset["y_test"],
                    batch_size=self.dataset["batch_size"],
                    verbose=0,
                    sample_weight=None
                    )

        except Exception as e:
            self.state["error"] = repr(e)
            return

        self.state["done"] = True


    def create_model(self):
        model = Sequential()
        last_layer = None

        layers = {
            "conv2d": add_layer_conv2d,
            "dense":  add_layer_dense,
            "pool":   add_layer_pool,
            "drop":   add_layer_drop,
        }

        multi_dimensional_layers = [
            "conv2d",
            "pool",
        ]

        for layer in self.state["network"]["layers"]:
            if (last_layer
                and last_layer["type"] in multi_dimensional_layers
                and layer["type"] not in multi_dimensional_layers):
                model.add(Flatten())
            layers[layer["type"]](model, layer, self.dataset, last_layer)
            last_layer = layer

        if self.state["network"]["layers"][-1]["type"] in ["conv2d", "pool"]:
            model.add(Flatten())

        model.add(Dense(self.dataset["nb_classes"]))
        model.add(Activation("softmax"))

        model.compile(loss=self.state["network"]["loss"],
                      optimizer='adadelta',
                      metrics=['accuracy'])

        self.state["loaded"] = True
        return model


    def sample_results(self, model):
        images = []
        predictions = model.predict_classes(self.dataset["x_test"], batch_size=128, verbose=0)
        for i in range(NUM_PREDICTIONS):
            random_selection = random.randrange(self.dataset["x_test"].shape[0])
            image_location = str(self.state["network"]["room"] + "_" + str(i))

            #TODO: support alpha channel
            if self.dataset["PNG_mode"] == "L":
                data_uri = png_l(
                    np.copy(self.dataset["x_test"][random_selection]),
                    self.dataset,
                    self.state["network"]["room"]
                )
                img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
            elif self.dataset["PNG_mode"] == "RGB":
                data_uri = png_rgb(
                    np.copy(self.dataset["x_test"][random_selection]),
                    self.dataset,
                    self.state["network"]["room"]
                )
                img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
            else:
                return {}

            predict = predictions.tolist()[random_selection]

            images.append({
                "expected": self.dataset["classes"][self.dataset["y_test"].tolist()[random_selection].index(1)],
                "predicted": self.dataset["classes"][predict],
                "img_tag": img_tag,
            })
        return images
