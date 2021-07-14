# @author : Abhishek R S

import os
import sys
import cv2
import argparse
import numpy as np
import tensorflow as tf
from scipy.misc import imread

tf.enable_eager_execution()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

color_maps = np.array([
    [0, 0, 0],  # other
    [128, 64, 128],  # road
    [244, 35, 232],  # sidewalk
    [250, 170, 30],  # traffic light
    [220, 220,  0],  # traffic sign
    [220, 20, 60],  # person
    [255,  0,  0],  # rider
    [0,  0, 142],  # car
    [0,  0, 70],  # truck
    [0, 60, 100],  # bus
    [0,  0, 90],  # caravan
    [0,  0, 110],  # trailer
    [0, 80, 100],  # train
    [0,  0, 230],  # motorcycle
    [119, 11, 32]],  # bicycle
    dtype=np.uint8)

def overlay_generator(FLAGS):
    images_list = os.listdir(FLAGS.images_dir)
    print(f"Number of overlays to generate : {len(images_list)}")

    if not os.path.exists(FLAGS.overlays_dir):
        os.makedirs(FLAGS.overlays_dir)

    for image_file in images_list:
        image = cv2.imread(os.path.join(FLAGS.images_dir, image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = imread(os.path.join(FLAGS.labels_dir, "label_" + image_file))

        label_one_hot = tf.one_hot(label, depth=len(color_maps), dtype=tf.uint8).numpy()
        mask = np.dot(label_one_hot, color_maps)

        image_mask_overlay = cv2.addWeighted(image, 1, mask, FLAGS.alpha, 0, image)
        image_mask_overlay = cv2.cvtColor(image_mask_overlay, cv2.COLOR_BGR2RGB)

        cv2.imwrite(os.path.join(FLAGS.overlays_dir, "overlay_" + image_file), image_mask_overlay)

def main():
    images_dir = "/opt/data/abhishek/cityscapes/resized_images/test/"
    labels_dir = "./model_fcn16_100/labels_100/"
    overlays_dir = "./model_fcn16_100/overlays_100/"
    alpha = 0.75

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--images_dir", default=images_dir,
        type=str, help="path to load image files")
    parser.add_argument("--labels_dir", default=labels_dir,
        type=str, help="path to load label files")
    parser.add_argument("--overlays_dir", default=overlays_dir,
        type=str, help="path to save image-mask overlay files")
    parser.add_argument("--alpha", default=alpha,
        type=str, help="alpha to control transparency of mask on overlay")

    FLAGS, unparsed = parser.parse_known_args()

    print("Generating overlays for images and masks")
    print("Overlay generation started...")
    overlay_generator(FLAGS)
    print("Overlay generation completed.")

if __name__ == "__main__":
    main()
