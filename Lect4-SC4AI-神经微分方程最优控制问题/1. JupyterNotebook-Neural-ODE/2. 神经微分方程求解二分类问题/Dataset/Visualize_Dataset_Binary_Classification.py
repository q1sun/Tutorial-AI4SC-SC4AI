import matplotlib.pyplot as plt
from tabulate import tabulate
from matplotlib.animation import FuncAnimation
import numpy as np


# display datasets
def plot_binary_classification_dataset(X, y, title=None, x_label=None, y_label=None, ax=None):
    CLASS_COLORS = ['coral', 'darkviolet']
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure
    ax.scatter(X[:, 0], X[:, 1], color=[CLASS_COLORS[yi.int()] for yi in y], s=5, alpha=0.6)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    return fig, ax

def print_train_test_dataloader(dataloader_train, dataloader_test):

    for batch_idx, (data, target) in enumerate(dataloader_train):
        datasize_train = data.shape
        datatype_train = data.dtype
        datasmpl_train = data[:2]
        datalabl_train = target[:2]

    train_info = {
        "fullbatch_size": len(dataloader_train.dataset),
        "minibatch_size": dataloader_train.batch_size, 
        "data_shape": f"{datasize_train} (type={datatype_train})", 
        "data_sample": datasmpl_train,
        "data_label": datalabl_train,
    }

    for batch_idx, (data, target) in enumerate(dataloader_test):
        datasize_test = data.shape
        datatype_test = data.dtype
        datasmpl_test = data[:2]
        datalabl_test = target[:2]

    test_info = {
        "fullbatch_size": len(dataloader_test.dataset),
        "minibatch_size": dataloader_test.batch_size,
        "data_shape": f"{datasize_test} (type={datatype_test})",
        "data_sample": datasmpl_test,
        "data_label": datalabl_test,
    }

    # convert to table for display
    table = []
    for key in train_info:
        table.append([key, train_info[key], test_info[key]])

    print(tabulate(
        table,
        headers=["", "Train DataLoader", "Test DataLoader"],
        tablefmt="fancy_grid"
    ))


# plot coordinate-wise trajectories through trained model, together with label prediction
def plot_trajectories(time_span, trajectories, label_trajectories, class_colors):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 4)) 

    for i in range(trajectories.shape[1]):
        ax0.plot(time_span, trajectories[:, i, 0], color=class_colors[i], alpha=0.1)
        ax1.plot(time_span, trajectories[:, i, 1], color=class_colors[i], alpha=0.1)
    ax0.set_xlabel(r"$t$")
    ax0.set_ylabel(r"$X_0(t)$")
    ax0.set_title("Evolution of $X_0(t)$")
    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$X_1(t)$")
    ax1.set_title("Evolution of $X_1(t)$")

    for i in range(label_trajectories.shape[1]):
        ax2.plot(time_span, label_trajectories[:, i, 0], color=class_colors[i], alpha=0.1)
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"label prediction")
    ax2.set_title("Evolution of Softmax[$X(t)$]")
    plt.subplots_adjust(wspace=0.3) 

def plot_trajectories_animation(time_span, trajectories, colors, classes, lim=10.0):
    def animate_frame(t):
        ax.cla()
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title('Evolution of Sample Points')
        ax.set_xlabel(r"$X_0(t)$")
        ax.set_ylabel(r"$X_1(t)$")

        zero_classes = np.array(classes) == 0
        one_classes = np.array(classes) == 1

        scatter_zero = ax.plot(
            trajectories[t, zero_classes, 0], trajectories[t, zero_classes, 1],
            'o', markersize=3, color=colors[0], alpha=0.2+0.8*t/len(time_span))
        scatter_one = ax.plot(
            trajectories[t, one_classes, 0], trajectories[t, one_classes, 1],
            'o', markersize=3, color=colors[1], alpha=0.2+0.8*t/len(time_span))
        return scatter_zero, scatter_one

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    anim = FuncAnimation(fig, animate_frame, frames=len(time_span))
    plt.close(fig)
    return anim