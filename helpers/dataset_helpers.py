from keras.utils import np_utils


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
