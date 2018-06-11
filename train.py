import time
import argparse
from datetime import datetime

import numpy as np
import tensorflow as tf
import cv2
from random import shuffle

from vae_original import VariationalAutoencoder

def load_images(binarize=True):
    count = 0
    im_per_class = 500
    num_images = 8 * 32 * im_per_class
    (x_size, y_size) = (30, 30)
    x = np.empty((num_images, y_size, x_size, 1), int)
    y = np.empty((num_images,), int)
    surfaces = [32, 64, 96, 128, 160, 192, 224, 256]
    for sum_surface in surfaces:
        for num_obj in range(1, 33):
            directory = "/home/dvanleeuwen/data/gen_img_3030/surf" + str(sum_surface) + "_obj"+ str(num_obj)
            for i in range(im_per_class):
                temp_data = cv2.imread(directory + "/image_" + str(sum_surface) + "_" + str(num_obj) + "_" + str(i) + ".png", 0)
                if temp_data is None:
                    print("could not load image: " + directory + "/image_" + str(sum_surface) + "_" + str(num_obj) + "_" + str(i) + ".png")
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
    surfaces = [32, 64, 96, 128, 160, 192, 224, 256]
    num_obj_list = list(range(1, 33))
    shuffle(surfaces)
    shuffle(num_obj_list)
    for sum_surface in surfaces:
        for num_obj in num_obj_list:
            directory = "/home/dvanleeuwen/data/gen_img_3030/surf" + str(sum_surface) + "_obj" + str(num_obj)
            for i in range(im_per_class):
                filenames.append(directory + "/image_" + str(sum_surface) + "_" + str(num_obj) + "_" + str(i) + ".png")
    test_size = 5000
    if int(len(filenames) * 0.9) < test_size:
        test_size = int(len(filenames) * 0.9)
    np.random.shuffle(filenames)
    train_filenames = filenames[test_size:]
    test_filenames = filenames[:test_size]
    return train_filenames, test_filenames


def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.cast(tf.image.decode_png(image_string, 1), tf.float32)
    return image_decoded

# def load_mnist_images(binarize=True):
#     from tensorflow.examples.tutorials.mnist import input_data
#     mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
#     x_train = mnist.train.images.reshape(-1, 28, 28, 1)
#     x_test  = mnist.test.images.reshape(-1, 28, 28, 1)
#     if binarize:
#         x_train = (x_train>0.5).astype(x_train.dtype)
#         x_test  = (x_test>0.5).astype(x_test.dtype)
#     return x_train, x_test

# show samples
def show_samples(sess, vae):
    im_samples = sess.run(vae.sample(100))
    for i in range(100):
        cv2.imshow("Example", np.squeeze(im_samples[i]*255, axis=-1).astype(np.uint8))
        if cv2.waitKey(0) == ord('q'):
            break


def train_vae(config):
    # save_path = "/models/model30.ckpt"
    save_path = config.model_path + "model" + config.model_name + ".ckpt"
    load_model = False
    save_model = True

    # Load dataset
    # x_train, x_test = load_images(binarize=True)
    # print(x_train.shape)
    # print(x_test.shape)
    # print("z_dim: %d  " % config.z_dim)
    # n_samples, im_height, im_width, _ = x_train.shape

    # Data pipeline
    train_filenames, test_filenames = get_image_filenames()

    # Data pipeline

    n_samples, im_height, im_width = (len(train_filenames), 30, 30)
    data_train = tf.data.Dataset.from_tensor_slices(train_filenames)
    # dataset = dataset.shuffle(len(filenames))
    data_train = data_train.map(_parse_function)
    # data_train = tf.data.Dataset.from_tensor_slices(x_train)
    # data_train = data_train.shuffle(buffer_size=10000)
    data_train = data_train.repeat(config.num_epochs)
    data_train = data_train.batch(config.batch_size)

    data_test = tf.data.Dataset.from_tensor_slices(test_filenames)
    data_test = data_test.map(_parse_function)
    data_test = data_test.batch(config.batch_size)
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, data_train.output_types, data_train.output_shapes)
    # iterator = tf.data.Iterator.from_structure(data_train.output_types, data_train.output_shapes)
    im_batch = tf.reshape(iterator.get_next(), [-1, im_height, im_width, 1])
    # init_train_data_op = iterator.make_initializer(data_train)
    # init_test_data_op  = iterator.make_initializer(data_test)

    iter_train = data_train.make_one_shot_iterator()
    iter_test = data_test.make_initializable_iterator()

    # Build VAE model
    model = VariationalAutoencoder(
        x_shape=(im_height, im_width),
        z_dim=config.z_dim,
    )

    global_step = tf.train.get_or_create_global_step()

    # Lower bound and loss
    print("x_minibatch", im_batch.shape)

    lower_bound = model.lower_bound(im_batch)
    loss = -tf.reduce_mean(lower_bound)

    ema = tf.train.ExponentialMovingAverage(decay=0.99, zero_debias=True)
    main_averages_op = ema.apply([loss])

    # Optimizer
    optimizer = tf.train.AdamOptimizer(config.learning_rate)
    apply_gradients_op = optimizer.minimize(loss, global_step)
    with tf.control_dependencies([apply_gradients_op]):
        train_op = tf.group(main_averages_op)

    # Build the code for plotting the manifold.
    z0, z1 = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
    z_grid_points = tf.constant(np.concatenate([z0.flatten()[:, None], z1.flatten()[:, None], np.zeros((5**2, config.z_dim-2))], axis=1).astype(np.float32))
    x_points = model.mean_x_given_z(z_grid_points)
    x_points = tf.reshape(x_points, shape=[-1, 30, 30, 1])

    # Sample from the model, for visualization
    samples = model.sample(10)
    samples = tf.reshape(samples, [-1,30,30,1])
    samples = tf.cast(samples, tf.float32)

    # Placeholder for test loss
    pl_test_loss = tf.placeholder(tf.float32, [], name="test_loss")

    # Initialize summaries
    train_scalar_summaries_op = tf.summary.merge([tf.summary.scalar("train/loss", loss)])
    test_scalar_summaries_op  = tf.summary.merge([tf.summary.scalar("test/loss", pl_test_loss)])

    image_summaries = [
        tf.summary.image("mean_x_given_z", x_points, max_outputs=30),
        tf.summary.image("samples", samples, max_outputs=30)
    ]
    image_summaries_op = tf.summary.merge(image_summaries)
    summary_writer = tf.summary.FileWriter(config.summary_path+"run_{}".format(datetime.now().strftime("%Y%m%d_%H%M")))

    # Initialize TensorFlow session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_mem_frac)
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=config.log_device_placement))

    # load model
    if load_model:
        saver.restore(sess, save_path)
        print("Model restored.")

    # sess.run(init_train_data_op) # Initialize the variables of the data-loader.
    sess.run(tf.global_variables_initializer())  # Initialize the model parameters.

    num_steps = int((config.num_epochs * n_samples) / config.batch_size)
    examples_per_second = 0

    train_handle = sess.run(iter_train.string_handle())
    test_handle = sess.run(iter_test.string_handle())

    train_ops = {
        'train_op': train_op,
        'train_lower_bound': tf.reduce_mean(lower_bound),
        'train_loss': ema.average(loss),
        'train_summary': train_scalar_summaries_op
    }

    for step in range(num_steps):

        epoch = (step*config.batch_size)/float(n_samples)
        t1 = time.time()

        # Training operation
        res = sess.run(train_ops, feed_dict={handle: train_handle})

        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % config.print_every == 0:
            print("[{}] Train Step {:04d}, Epoch {:.1f}, Batch Size = {}, Examples/Sec = {:.2f}, Train LB = {:.3f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step, epoch,
                    config.batch_size, examples_per_second, res['train_lower_bound'], res['train_loss']))

        # Trainings summary with scalars
        summary_writer.add_summary(res['train_summary'], step)

        # Sample from the model less frequently
        if step % config.sample_every == 0 and step > 0:
            image_summary_str = sess.run(image_summaries_op)
            summary_writer.add_summary(image_summary_str, step)

        if step % config.test_every == 0 and step > 0:

            # Switch to test data
            # sess.run(init_test_data_op)
            sess.run(iter_test.initializer)

            test_losses, test_lower_bounds = [], []
            while True:
                try:
                    test_loss, test_lower_bound = sess.run([loss, lower_bound], feed_dict={handle: test_handle})
                    test_losses.append(test_loss)
                    test_lower_bounds.append(np.mean(test_lower_bound))
                except tf.errors.OutOfRangeError:
                    break

            print("Performance on test set:")
            print("  Test Lower Bound = {:.03f}, Test Loss = {:.03f}".format(np.mean(test_lower_bounds), np.mean(test_losses)))

            # Write test summaries to disk
            test_summary_str = sess.run(test_scalar_summaries_op, feed_dict={pl_test_loss: np.mean(test_losses)})
            summary_writer.add_summary(test_summary_str, step)

            # Switch to train data
            # sess.run(init_train_data_op)
    if save_model:
        save_path = saver.save(sess, save_path)
        print("Model saved in path: %s" % save_path)
    # show some generated samples
    # show_samples(sess, model)


if __name__ == '__main__':

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--z_dim', type=int, default=20, help='Dimensionality of z')#defaul=2

    # Training params
    parser.add_argument('--batch_size', type=int, default=256, help='Number of examples to process in a batch')#default=100
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')#default=0.003
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')#default=1000

    # Print, sampling and testing frequency
    parser.add_argument('--print_every', type=int, default=25, help='Frequency of printing model train performance')
    parser.add_argument('--sample_every', type=int, default=1000, help='Frequency of sampling from model')#default=500
    parser.add_argument('--test_every', type=int, default=200, help='Frequency of testing model performance')

    # Misc params
    parser.add_argument('--gpu_mem_frac', type=float, default=0.7, help='Fraction of GPU memory to allocate')#default=0.5
    parser.add_argument('--log_device_placement', type=bool, default=False, help='Log device placement for debugging')
    parser.add_argument('--summary_path', type=str, default="/home/dvanleeuwen/data/logs/", help='Output path for summaries')#default="./summaries/"
    parser.add_argument('--model_path', type=str, default="/home/dvanleeuwen/data/models/", help='Output path for the model')
    parser.add_argument('--model_name', type=str, default="3030", help='Output name for the model')

    config = parser.parse_args()

train_vae(config)