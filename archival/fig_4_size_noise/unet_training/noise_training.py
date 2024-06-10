import numpy as np
import pandas as pd
import segmentation_models as sm
import h5py

from tqdm import tqdm
from itertools import product
import pathlib
import datetime
import json

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import backend as K


def is_hidden(path):
    for x in str(path).split("/"):
        if x.startswith(".") and x != "..":
            return True

    return False


def listfiles(folder, include_hidden=False):
    # generator for files in subdirectory

    if include_hidden:
        out = [x for x in pathlib.Path(folder).glob("**/*")]
        return out
    else:
        out = [x for x in pathlib.Path(folder).glob("**/*") if not is_hidden(x)]
        return out


def gen_folder_name(base="particle"):
    today = datetime.today()
    formatToday = today.strftime("_%Y%m%d")
    hTime = int(today.strftime("%H"))
    mTime = int(today.strftime("%M"))
    sTime = int(today.strftime("%S"))
    fTime = today.strftime("%f")

    # convert time into seconds
    time = (hTime * 3600) + (mTime * 60) + sTime
    time = str(time)

    # add together for file name
    folderName = base + "_" + time + fTime + formatToday
    return folderName


def get_new_folder(mkdir=True, base="particle"):
    new_folder = gen_folder_name(base)
    while pathlib.Path(new_folder).is_dir():
        new_folder = gen_folder_name(base)

    if mkdir:
        pathlib.Path(new_folder).mkdir(parents=True)

    return pathlib.Path(new_folder)


def orthogonal_rot(image):
    """Preprocessing function to limit rotations to 0, 90, 180, 270

    based on https://stackoverflow.com/a/61304850/10094666
    """
    return np.rot90(image, np.random.choice([0, 1, 2, -1]))


def get_scheduler(factor):
    def scheduler(epoch, lr):
        return lr * factor

    return scheduler


def write_metadata(metadata, path):
    with open(str(path), "w") as f:
        json_string = json.dumps(metadata, sort_keys=True, indent=4)
        f.write(json_string)

    return


def read_metadata(path):
    f = open(path)
    metadata = json.load(f)
    f.close()
    return metadata


def main():
    rng = np.random.default_rng()

    # params include
    batch_size = 16
    target_dose = list(np.arange(100, 1300, 100))
    backbone = ["resnet18"]
    alpha_0 = [0.001]  # starting learning rate
    schedule = [(0.8, "every_epoch")]

    param_list = list(product(*(target_dose, backbone, alpha_0, schedule)))
    for p in tqdm(param_list):
        for i in range(5):
            seed = int(rng.integers(0, 1e6))
            params = {
                "batch_size": batch_size,
                "target_dose": p[0],
                "backbone": p[1],
                "alpha_0": p[2],
                "schedule": p[3],
                "seed": seed,
            }
            try:
                run_training(**params)
                K.clear_session()
            except:
                print("Network training failed for following parameters: ", params)
                print("Training failed on iteration %i" % i)


def run_training(**kwargs):
    # load data
    root_folder = "path/to/noise/data"

    folders = [x for x in listfiles(root_folder) if x.is_dir()]

    metadata_list = []
    for f in folders:
        dataX_path = f.joinpath("train.npy")
        dataY_path = f.joinpath("mask.npy")
        meta_dict = read_metadata(f.joinpath("metadata.json"))
        meta_dict["dataX_path"] = dataX_path
        meta_dict["dataY_path"] = dataY_path
        metadata_list.append(meta_dict)

    df = pd.DataFrame(metadata_list)

    target_dose = kwargs["target_dose"]
    df = df[df["dose"] == target_dose]
    df = df.drop_duplicates(subset="particle_ID")
    print("Data aggregated. Loading arrays into memory.")

    X = np.array([np.load(f) for f in df["dataX_path"]])
    X_min = np.min(X, axis=(1, 2))[:, None, None]
    X_max = np.max(X, axis=(1, 2))[:, None, None]
    X = (X - X_min) / (X_max - X_min)
    X = np.expand_dims(X, axis=3)

    Y = np.array([np.load(f) for f in df["dataY_path"]]).astype(np.float32)
    Y = np.expand_dims(Y, axis=3)
    Y = np.array(np.concatenate((np.abs(Y - 1), Y), axis=3))

    print("Data loaded and prepared.")
    print("X shape and memory footprint (Gb): (%i, %i, %i, %i), %f" % (*X.shape, X.nbytes / 1e9))
    print("Y shape and memory footprint (Gb): (%i, %i, %i, %i), %f" % (*Y.shape, Y.nbytes / 1e9))

    # prepare data generators
    seed = kwargs["seed"]
    data_gen_args = {
        "preprocessing_function": orthogonal_rot,
        "horizontal_flip": True,
        "vertical_flip": True,
        "validation_split": 0.25,
    }

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_datagen.fit(X, augment=True, seed=seed)
    mask_datagen.fit(Y, augment=True, seed=seed)

    batch_size = kwargs["batch_size"]

    steps_per_epoch = X.shape[0] * (1 - data_gen_args["validation_split"]) // batch_size
    validation_steps = X.shape[0] * (data_gen_args["validation_split"]) // batch_size

    image_generator_train = image_datagen.flow(X, batch_size=batch_size, seed=seed, subset="training")
    image_generator_test = image_datagen.flow(X, batch_size=batch_size, seed=seed, subset="validation")

    mask_generator_train = mask_datagen.flow(Y, batch_size=batch_size, seed=seed, subset="training")
    mask_generator_test = mask_datagen.flow(Y, batch_size=batch_size, seed=seed, subset="validation")

    train_generator = (
        pair for pair in zip(image_generator_train, mask_generator_train)
    )  # https://stackoverflow.com/a/65731446/10094666
    test_generator = (pair for pair in zip(image_generator_test, mask_generator_test))

    # prepare checkpointing and monitoring
    backbone = kwargs["backbone"]
    model_dir = get_new_folder(base="trained_models/unet")
    weights_path = model_dir.joinpath("sm_unet_noPretrainWeights_" + backbone + "_weights.h5")
    final_weights_path = model_dir.joinpath("sm_unet_noPretrainWeights_" + backbone + "_final_weights.h5")
    history_path = model_dir.joinpath("sm_unet_noPretrainWeights_" + backbone + "_history.h5")

    modelCheckpoint = ModelCheckpoint(
        weights_path, monitor="val_loss", save_best_only=True, mode="min", verbose=2, save_weights_only=True
    )

    callbacks_list = [modelCheckpoint]

    lr_schedule = kwargs["schedule"]
    if lr_schedule[1] == "every_epoch":
        schedule = LearningRateScheduler(get_scheduler(lr_schedule[0]))
    elif lr_schedule[1] == "on_plateau":
        schedule = ReduceLROnPlateau(monitor="val_loss", factor=lr_schedule[0], patience=5, mode="min")

    callbacks_list.append(schedule)

    # load model
    model = sm.Unet(backbone, encoder_weights=None, classes=2, activation="softmax", input_shape=(None, None, 1))
    learning_rate = kwargs["alpha_0"]
    model.compile(
        Adam(learning_rate=learning_rate),
        loss=sm.losses.cce_dice_loss,
        metrics=[sm.metrics.iou_score, sm.metrics.f1_score],
    )

    # train with history
    N_epochs = 25
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=N_epochs,
        callbacks=callbacks_list,
        validation_data=test_generator,
        validation_steps=validation_steps,
        verbose=2,
    )

    # save final weights, training history, metadata
    write_metadata(kwargs, model_dir.joinpath("metadata.json"))
    model.save_weights(final_weights_path)
    h = h5py.File(history_path, "w")
    h_keys = history.history.keys()
    print(h_keys)
    for k in h_keys:
        h.create_dataset(k, data=history.history[k])
    h.close()


if __name__ == "__main__":
    main()
