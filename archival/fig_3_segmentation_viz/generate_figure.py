import segmentation_models as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import h5py
import seaborn as sns
import re

import json

from functools import reduce
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


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


def read_metadata(path):
    f = open(path)
    metadata = json.load(f)
    f.close()
    return metadata


def read_training_history(path):
    history_file = h5py.File(path, "r")
    history_dict = {x: np.array(y) for x, y in history_file.items()}
    history_file.close()
    return history_dict


def grab_model_metadata():
    root_dir_base = pathlib.Path("/path/to/base/folder")
    root_dirs = [
        root_dir_base.joinpath("baseline_models"),
        root_dir_base.joinpath("substrate_varying_models"),
        root_dir_base.joinpath("optimal_Au_models"),
    ]

    root_dirs = [folder.joinpath("trained_models") for folder in root_dirs]
    model_folders = [[x for x in listfiles(r) if x.is_dir()] for r in root_dirs]
    model_folders = reduce(lambda x, y: x + y, model_folders)

    ## load all the metadata from the networks
    metadata_list = []
    for folder in model_folders:
        model_metadata = read_metadata(folder.joinpath("metadata.json"))
        model_id = re.search("[0-9]+_[0-9]+", str(folder).rsplit("/")[-1])[0]
        model_metadata["ID"] = model_id
        if "transfer" in str(folder):
            tl = True
            history_path = "sm_unet_transferLearnWeights_" + model_metadata["backbone"] + "_history.h5"
            history_dict = read_training_history(folder.joinpath(history_path))
            model_metadata.update(history_dict)
            model_metadata["series"] = "transfer"
        else:
            tl = False
            model_metadata["series"] = str(folder).rsplit("/")[-3].split("_")[0]
        model_metadata["transfer_learned"] = tl
        model_metadata["folder"] = folder
        metadata_list.append(model_metadata)

    df = pd.DataFrame(metadata_list)
    df = pd.concat([df, pd.DataFrame(df["schedule"].to_list(), columns=["schedule_rate", "schedule_timing"])], axis=1)

    return df


def load_experimental_data():
    expt_data_path = "/path/to/data"
    expt_mask_path = "/path/to/labels"

    f_tmp = h5py.File(expt_data_path, "r")
    X = np.array(f_tmp["images"]).astype(np.float32)
    f_tmp.close()

    f_tmp = h5py.File(expt_mask_path, "r")
    Y = np.array(f_tmp["maps"]).astype(np.float32)
    f_tmp.close()

    return X, Y


def select_models(df):
    # Select models via quantile based performance on target metric
    metric_list = ["exp_f1-score"]
    q_list = [0.50, 0.65]
    models = []

    filt_b = df["series"] == "baseline"
    filt_s = df["series"] == "substrate"
    filt_o = df["series"] == "optimal"

    f_df = df[filt_b]
    for m in metric_list:
        for q in q_list:
            cur_ID = f_df[np.abs(f_df[m] - f_df.quantile(q=q, interpolation="nearest")[m]) < 0.0001].iloc[0]
            models.append(cur_ID)

    f_df = df[filt_s]
    q_list = [0.95]
    for m in metric_list:
        for q in q_list:
            cur_ID = f_df[np.abs(f_df[m] - f_df.quantile(q=q, interpolation="nearest")[m]) < 0.0001].iloc[0]
            models.append(cur_ID)

    f_df = df[filt_o]
    for m in metric_list:
        for q in q_list:
            cur_ID = f_df[np.abs(f_df[m] - f_df.quantile(q=q, interpolation="nearest")[m]) < 0.0001].iloc[0]
            models.append(cur_ID)


def get_predictions(model_metadata, X, indices):
    model = sm.Unet("resnet18", encoder_weights=None, classes=2, activation="softmax", input_shape=(None, None, 1))
    model.compile()

    predictions = []
    for m in model_metadata:
        model.load_weights(m["folder"].joinpath("sm_unet_noPretrainWeights_resnet18_weights.h5"))
        current_predictions = []
        for i in indices:
            current_predictions.append(m.predict(np.expand_dims(X[i, :, :, :], axis=0)))

        predictions.append(current_predictions)
    return predictions


def make_figure(model_metadata, X, Y, predictions, indices, outfile=None):
    fig, axes = plt.subplots(5, 4, figsize=(10, 12.5), facecolor="w")

    cmap_data = sns.color_palette("gray", as_cmap=True)
    cmap_predictions = sns.color_palette("coolwarm_r", as_cmap=True)
    cmap_truth = sns.color_palette("coolwarm", as_cmap=True)

    axes[0, 0].matshow(X[indices[0], :, :, 0], cmap=cmap_data)
    axes[0, 1].matshow(X[indices[1], :, :, 0], cmap=cmap_data)
    axes[0, 2].matshow(X[indices[2], :, :, 0], cmap=cmap_data)
    axes[0, 3].matshow(X[indices[3], :, :, 0], cmap=cmap_data)

    for j in range(4):
        axes[0, j].matshow(Y[indices[j], :, :, 1], cmap=cmap_truth, alpha=0.25, vmin=-1, vmax=1)

    cmap_predictions = sns.color_palette("coolwarm_r", as_cmap=True)
    cmap_truth = sns.color_palette("coolwarm", as_cmap=True)
    alpha_red = 0.9
    alpha_blue = 0.8

    for i, images in enumerate(predictions):
        for j, img in enumerate(images):
            axes[i + 1, j].matshow(Y[indices[j], :, :, 1], cmap=cmap_truth, alpha=alpha_red, vmin=-1, vmax=1)
            axes[i + 1, j].matshow(img[0, :, :, 1], cmap=cmap_predictions, alpha=alpha_blue, vmin=-1, vmax=1)

    for i, m in enumerate(model_metadata):
        axes[i + 1, 0].set_ylabel("Avg. F1-score: %0.4f" % m["exp_f1-score"])

    rs = AnchoredSizeBar(
        axes[-1, 0].transData,
        5 / (2 * 0.02152),
        "5.0 nm",
        "lower left",
        pad=0.2,
        color="black",
        frameon=False,
        label_top=True,
        size_vertical=3,
    )

    axes[-1, 0].add_artist(rs)

    for i in range(4):
        if i == 0:
            color = "white"
        else:
            color = "black"
        rs = AnchoredSizeBar(
            axes[i, 0].transData,
            2.5 / (0.02152),
            "     ",
            "lower left",
            pad=0.2,
            color=color,
            frameon=False,
            label_top=True,
            size_vertical=3,
        )
        axes[i, 0].add_artist(rs)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        plt.setp(ax.spines.values(), linewidth=1.25)

    if outfile is not None:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    df = grab_model_metadata()
    model_metadata = select_models(df)
    X, Y = load_experimental_data()
    indices = [0, 1, 2, 3, 4]  # arbitrary choice
    predictions = get_predictions(model_metadata, X, indices)
    make_figure(model_metadata, X, Y, predictions, indices)
