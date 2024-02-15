import os
import random

import numpy as np
import torch

import torchvision.io as io
import torchvision.transforms.v2.functional as tf

from distutils import dir_util
from matplotlib import pyplot as plt, gridspec

from train import mask_to_rgb_dict

def plot_loss(H, title="Graph", up_to=-1, save_name=""):
    fig, ax = plt.subplots()
    ax.plot(H["train_loss"][:up_to], label="Training loss", color='blue')
    ax.plot(H["eval_loss"][:up_to], label="Validation loss", color='red')
    ax.set_title(title)
    ax.set_xlabel("Epoch #")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    if save_name:
        plt.savefig(os.path.join("img", save_name + '.png'), bbox_inches="tight")
    plt.show()

def mean_metric(model, dataset, metric, downsize_img=False):
    total = 0

    with torch.no_grad():
        for (x, y) in dataset:
            x = normalize(x)
            if downsize_img:
                x = downsize(x)
                y = y.unsqueeze(0)
                y = downsize(y)
                y = y.squeeze(0)
            pred = model(x.unsqueeze(0))
            pred = torch.argmax(pred, dim=1)
            pred = pred.squeeze(0)

            total += metric(pred, y)

    return total.item() / len(dataset)

def mean_metric_per_class(model, dataset, metric, downsize_img=False):
    total = 0

    with torch.no_grad():
        for (x, y) in dataset:
            x = normalize(x)
            if downsize_img:
                x = downsize(x)
                y = y.unsqueeze(0)
                y = downsize(y)
                y = y.squeeze(0)
            pred = model(x.unsqueeze(0))
            pred = torch.argmax(pred, dim=1)
            pred = pred.squeeze(0)

            total += metric(pred, y)

    return total / len(dataset)

def print_metrics(metrics):
    print("Mean IOU:", metrics[0])
    print("Mean Precision:", metrics[1])
    print("Mean Recall:", metrics[2])
    print("Mean F1:", metrics[3])

"""
def visualize_tensor_mask (input_tensor):
    img = input_tensor.detach().numpy()#.transpose(1, 2, 0).astype(np.uint8)
    rgb = np.zeros((img.shape[0], img.shape[1], 3)).astype(np.uint8)
    for class_code, rgb_code in mask_to_rgb_dict.items():
        rgb[img == class_code, :] = rgb_code
    #rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)

def visualize(input_tensor, to_window=False):
    am = torch.argmax(input_tensor, dim=1)
    am = am.squeeze(0)
    img = am.numpy()#.transpose(1, 2, 0).astype(np.uint8)
    print(np.unique(img))
    rgb = np.zeros((img.shape[0], img.shape[1], 3)).astype(np.uint8)
    for class_code, rgb_code in mask_to_rgb_dict.items():
        rgb[img == class_code, :] = rgb_code
    #rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    if to_window:
        cv2.imshow("window", rgb)
        cv2.waitKey(0)
    else:
        plt.imshow(rgb)
"""

def visualize_four(x1, x2, x3, x4, y1, y2, y3, y4, pred1, pred2, pred3, pred4, title="Visualizations", save_name=""):
    fig, axs = plt.subplots(3, 4, constrained_layout=True)
    fig.suptitle(title)
    fig.text(-0.02, 0.7, "Sample map", rotation=90)
    fig.text(-0.02, 0.38, "Ground truth", rotation=90)
    fig.text(-0.02, 0.05, "Model prediction", rotation=90)
    for i, x in enumerate([x1, x2, x3, x4]):
        ax = axs[0][i]
        ax.imshow(x.permute(1, 2, 0))
        ax.axis('off')
        ax.set_ylabel('hi')
    for i, y in enumerate([y1, y2, y3, y4]):
        ax = axs[1][i]
        y = y.detach().numpy()
        rgb = np.zeros((*(y.shape), 3)).astype(np.uint8)
        for class_code, rgb_code in mask_to_rgb_dict.items():
            rgb[y == class_code, :] = rgb_code
        ax.imshow(rgb)
        ax.axis('off')
    for i, pred in enumerate([pred1, pred2, pred3, pred4]):
        ax = axs[2][i]
        pred = torch.argmax(pred, dim=1)
        pred = pred.squeeze(0)
        pred = pred.numpy()
        rgb = np.zeros((*(pred.shape), 3)).astype(np.uint8)
        for class_code, rgb_code in mask_to_rgb_dict.items():
            rgb[pred == class_code, :] = rgb_code
        ax.imshow(rgb)
        ax.axis('off')
    if save_name:
        plt.savefig(os.path.join("img", save_name + '.png'), bbox_inches="tight")

def normalize(img):
    mean = img.mean([1, 2])
    std = img.std([1, 2])
    return tf.normalize(img, mean, std)

def downsize(img):
    return tf.resize(img, img.shape[-1] // 2)

def data_augmentation(img_size: int = 1000):

    original_path = os.path.join('data', 'img')
    augmented_path = os.path.join('data', 'augmented')

    dir_util.remove_tree(augmented_path)
    dir_util.copy_tree(original_path, augmented_path)

    for dataset in os.listdir(augmented_path):
        print(dataset)
        img_dir = os.path.join(augmented_path, dataset, "train", "img")
        label_dir = os.path.join(augmented_path, dataset, "train", "label")

        img_filenames = [filename for filename in os.listdir(img_dir) if filename.endswith('.png')]
        label_filenames = [filename for filename in os.listdir(label_dir) if filename.endswith('.png')]

        for (img_filename, label_filename) in zip(img_filenames, label_filenames):
            if img_filename.endswith('.png') and label_filename.endswith('.png'):
                img_path = os.path.join(img_dir, img_filename)
                img = io.read_image(img_path)
                img_name = img_path.split('.')[0]

                label_path = os.path.join(label_dir, label_filename)
                label = io.read_image(label_path)
                label_name = label_path.split('.')[0]

                print(img_path)

                # Vflip
                img_tf = tf.functional.vflip(img)
                io.write_png(img_tf, img_name + "_vflip" + ".png")
                label_tf = tf.functional.vflip(label)
                io.write_png(label_tf, label_name + "_vflip" + ".png")

                # Hflip
                img_tf = tf.functional.hflip(img)
                io.write_png(img_tf, img_name + "_hflip" + ".png")
                label_tf = tf.functional.hflip(label)
                io.write_png(label_tf, label_name + "_hflip" + ".png")

                # Center crop
                img_tf = tf.functional.resized_crop(img, img_size // 4, img_size // 4, img_size // 2, img_size // 2, img_size, interpolation=tf.InterpolationMode.NEAREST)
                io.write_png(img_tf, img_name + "_center_crop" + ".png")
                label_tf = tf.functional.resized_crop(label, img_size // 4, img_size // 4, img_size // 2, img_size // 2, img_size, interpolation=tf.InterpolationMode.NEAREST)
                io.write_png(label_tf, label_name + "_center_crop" + ".png")

                # Random crop
                params = tf.RandomResizedCrop.get_params(img, scale=(0.5, 1.0), ratio=(1.0, 1.0))
                img_tf = tf.functional.resized_crop(img, *params, size=img_size, interpolation=tf.InterpolationMode.NEAREST)
                io.write_png(img_tf, img_name + "_random_crop" + ".png")
                label_tf = tf.functional.resized_crop(label, *params, size=img_size, interpolation=tf.InterpolationMode.NEAREST)
                io.write_png(label_tf, label_name + "_random_crop" + ".png")

                # Random erasing
                params = tf.RandomErasing.get_params(img, scale=(0.05, 0.20), ratio=(1.0, 1.0), value=[0])
                img_tf = tf.functional.erase(img, *params)
                io.write_png(img_tf, img_name + "_random_erase" + ".png")
                label_tf = tf.functional.erase(label, *params)
                io.write_png(label_tf, label_name + "_random_erase" + ".png")

                # Random color jitter
                random_tf = tf.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.2, hue=0.1)
                img_tf = random_tf(img)
                io.write_png(img_tf, img_name + "_color_jitter_1" + ".png")
                io.write_png(label, label_name + "_color_jitter_1" + ".png")

                random_tf = tf.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.1, hue=0.1)
                img_tf = random_tf(img)
                io.write_png(img_tf, img_name + "_color_jitter_2" + ".png")
                io.write_png(label, label_name + "_color_jitter_2" + ".png")

                random_tf = tf.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.1)
                img_tf = random_tf(img)
                io.write_png(img_tf, img_name + "_color_jitter_3" + ".png")
                io.write_png(label, label_name + "_color_jitter_3" + ".png")

                random_tf = tf.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.3, hue=0.3)
                img_tf = random_tf(img)
                io.write_png(img_tf, img_name + "_color_jitter_4" + ".png")
                io.write_png(label, label_name + "_color_jitter_4" + ".png")


if __name__ == "__main__":
    seed = 343

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_augmentation()