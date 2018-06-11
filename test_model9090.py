import time
import argparse
from datetime import datetime

import numpy as np
import tensorflow as tf
import cv2

from vae9090 import VariationalAutoencoder
from random import shuffle

import sys

# num_images = 8*32*200
# (x_size, y_size) = (30, 30)
# data = np.empty((num_images, y_size, x_size, 1), int)
# data_y = np.empty((num_images,), int)


# def load_images(directory, show_load=False):
#     count = 0
#     for subdir, dirs, files in os.walk(directory):
#         for file in files:
#             # print os.path.join(subdir, file)
#             # print(file)
#             if file.endswith(".png"):
#                 temp_data = cv2.imread(os.path.join(subdir, file), 0)
#                 # data[count] = np.expand_dims(cv2.resize(temp_data, (28, 28)), -1)
#                 data[count] = np.expand_dims(temp_data, -1)
#                 # f_strings.append(os.path.join(subdir, file))
#                 data_y[count] = int(file.split("_")[2])
#                 count += 1
#             if show_load and count % (num_images / 10) == 0:
#                 print(str(count / (num_images / 10) * 10) + "%")
#     return

def load_images(binarize=True):
    count = 0
    surfaces = [i * 9 for i in [32, 64, 96, 128, 160, 192, 224, 256]]
    object_nums = range(0, 20)
    im_per_class = 500
    num_images = len(surfaces) * len(object_nums) * im_per_class
    (x_size, y_size) = (90, 90)
    x = np.empty((num_images, y_size, x_size, 1), int)
    y = np.empty((num_images,), int)
    for sum_surface in surfaces:
        for num_obj in object_nums:
            directory = "/home/dvanleeuwen/data/gen_img_9090/surf" + str(sum_surface) + "_obj"+ str(num_obj)
            for i in range(im_per_class):
                temp_data = cv2.imread(directory + "/image_" + str(sum_surface) + "_" + str(num_obj) + "_" + str(i) + ".png", 0)
                x[count] = np.expand_dims(temp_data, -1)
                y[count] = num_obj
                count += 1
    np.random.shuffle(x)
    x_train, x_test = x[:int(num_images*9/10),:].astype(np.float32), x[int(num_images*9/10):,:].astype(np.float32)
    if binarize:
        x_train = (x_train>0.5).astype(x_test.dtype)
        x_test  = (x_test>0.5).astype(x_test.dtype)
    return x_train, x_test

def get_image_filenames():
    im_per_class = 5000
    filenames = []
    surfaces = [i * 9 for i in [32, 64, 96, 128, 160, 192, 224, 256]]
    num_obj_list = list(range(0, 21))
    shuffle(surfaces)
    shuffle(num_obj_list)
    for sum_surface in surfaces:
        for num_obj in num_obj_list:
            directory = "/home/dvanleeuwen/data/gen_img_9090/surf" + str(sum_surface) + "_obj" + str(num_obj)
            for i in range(im_per_class):
                filenames.append(directory + "/image_" + str(sum_surface) + "_" + str(num_obj) + "_" + str(i) + ".png")
    np.random.shuffle(filenames)
    return filenames


def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.cast(tf.image.decode_png(image_string, 1), tf.float32)
    return image_decoded


def train_vae(config):
    # save_path = "/models/model30.ckpt"
    save_path = config.model_path + "model" + config.model_name + ".ckpt"
    load_model = True

    # Load dataset
    filenames = get_image_filenames()

    # Data pipeline
    n_samples, im_height, im_width = (len(filenames), 90, 90)
    data = tf.data.Dataset.from_tensor_slices(filenames)
    data = data.map(_parse_function)

    data = data.repeat(config.num_epochs)
    data = data.batch(config.batch_size)

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, data.output_types, data.output_shapes)
    im_batch = tf.reshape(iterator.get_next(), [-1, im_height, im_width, 1])

    iter = data.make_one_shot_iterator()

    # Build VAE model
    model = VariationalAutoencoder(
        x_shape=(im_height, im_width),
        z_dim=config.z_dim,
    )

    # Lower bound and loss
    print("x_minibatch", im_batch.shape)
    test_code = model.test_coder(im_batch)

    # Initialize TensorFlow session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_mem_frac)
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=config.log_device_placement))

    # load model
    if load_model:
        saver.restore(sess, save_path)
        print("Model restored.")

    iter_handle = sess.run(iter.string_handle())

    while True:
        (original, decoded) = sess.run(test_code, feed_dict={handle: iter_handle})
        for i in range(config.batch_size):
            im1 = np.squeeze(original[i] * 255, axis=-1).astype(np.uint8)
            im2 = np.squeeze(decoded[i] * 255, axis=-1).astype(np.uint8)
            cv2.imshow("original", np.concatenate((im1, im2), axis=1))
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        else:
            continue
        break


if __name__ == '__main__':
    print("training...")

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--z_dim', type=int, default=50, help='Dimensionality of z')#defaul=2

    # Training params
    parser.add_argument('--batch_size', type=int, default=32, help='Number of examples to process in a batch')#default=100
    parser.add_argument('--learning_rate', type=float, default=0.003, help='Learning rate')#default=0.003
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of training epochs')#default=1000

    # Print, sampling and testing frequency
    parser.add_argument('--print_every', type=int, default=25, help='Frequency of printing model train performance')
    parser.add_argument('--sample_every', type=int, default=1000, help='Frequency of sampling from model')#default=500
    parser.add_argument('--test_every', type=int, default=200, help='Frequency of testing model performance')

    # Misc params
    parser.add_argument('--gpu_mem_frac', type=float, default=0.8, help='Fraction of GPU memory to allocate')#default=0.5
    parser.add_argument('--log_device_placement', type=bool, default=False, help='Log device placement for debugging')
    parser.add_argument('--summary_path', type=str, default="/home/dvanleeuwen/data/logs/", help='Output path for summaries')#default="./summaries/"
    parser.add_argument('--model_path', type=str, default="/home/dvanleeuwen/data/models/", help='Output path for the model')
    parser.add_argument('--model_name', type=str, default="9090_0_20", help='Output name for the model')

    config = parser.parse_args()

train_vae(config)