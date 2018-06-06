import numpy as np
import tensorflow as tf
import time
import argparse
from datetime import datetime

import cv2
import os


# Code based on vae_conv.py by Tom Runia

def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)


class VariationalAutoencoder(object):

    def __init__(self, x_shape, z_dim):
        self.kernel_initializer = tf.initializers.variance_scaling()
        self.x_shape = x_shape
        self.z_dim = z_dim

    def encode(self, x, is_train=True):
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):

            # ConvNet
            # 1st hidden layer
            conv1 = tf.layers.conv2d(x, 32, [4, 4], strides=(2, 2), padding='same')
            # conv1 = tf.Print(conv1, [tf.shape(conv1)], message="conv1: ", summarize=10)
            lrelu1 = lrelu(conv1, 0.2)

            # 2nd hidden layer
            conv2 = tf.layers.conv2d(lrelu1, 64, [4, 4], strides=(2, 2), padding='same', kernel_initializer=self.kernel_initializer)
            # conv2 = tf.Print(conv2, [tf.shape(conv2)], message="conv2: ", summarize=10)
            lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=is_train), 0.2)

            # 3rd hidden layer
            conv3 = tf.layers.conv2d(lrelu2, 64, [4, 4], strides=(2, 2), padding='same', kernel_initializer=self.kernel_initializer)
            # conv3 = tf.Print(conv3, [tf.shape(conv3)], message="conv3: ", summarize=10)
            lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=is_train), 0.2)

            # 4th hidden layer
            conv4 = tf.layers.conv2d(lrelu3, 128, [4, 4], strides=(2, 2), padding='same',
                                     kernel_initializer=self.kernel_initializer)
            # conv4 = tf.Print(conv4, [tf.shape(conv4)], message="conv4: ", summarize=10)
            lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=is_train), 0.2)

            # 4th hidden layer
            conv5 = tf.layers.conv2d(lrelu4, 128, [4, 4], strides=(2, 2), padding='same', kernel_initializer=self.kernel_initializer)
            # conv5 = tf.Print(conv5, [tf.shape(conv5)], message="conv5: ", summarize=10)
            lrelu5 = lrelu(tf.layers.batch_normalization(conv5, training=is_train), 0.2)
            lrelu5_flat = tf.layers.flatten(lrelu5)

            # Intermediate dense layer
            dense = tf.layers.dense(lrelu5_flat, 50, activation=tf.nn.relu, kernel_initializer=self.kernel_initializer)
            # dense = tf.Print(dense, [tf.shape(dense)], message="dense: ", summarize=10)

            # Parameters of Gaussian distribution
            z_gaussian_params = tf.layers.dense(dense, units=2*self.z_dim, kernel_initializer=self.kernel_initializer)

            # Mean parameter of Gaussian is unconstrained
            z_mean = z_gaussian_params[:,0:self.z_dim]

            # Standard deviation of Gaussian must be positive. Therefore, we
            # use a softplus and also add a small epsilon for numerical stability.
            z_sigma = 1e-6 + tf.nn.softplus(z_gaussian_params[:,self.z_dim:])

        return z_mean, z_sigma

    def decode(self, z, is_train=True):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):

            dense = tf.layers.dense(z, units=50, kernel_initializer=self.kernel_initializer, activation=tf.nn.relu)
            # dense = tf.Print(dense, [tf.shape(dense)], message="dense: ", summarize=10)
            decoder_expand = tf.layers.dense(dense, units=(128*4*4), kernel_initializer=self.kernel_initializer, activation=tf.nn.relu)
            decoder_expand = tf.reshape(decoder_expand, [-1,4,4,128])
            # decoder_expand = tf.Print(decoder_expand, [tf.shape(decoder_expand)], message="decoder_expand: ", summarize=10)

            # 1st hidden layer
            conv1 = tf.layers.conv2d_transpose(decoder_expand, 128, [4, 4], strides=(2, 2), padding='valid', kernel_initializer=self.kernel_initializer)
            # conv1 = tf.Print(conv1, [tf.shape(conv1)], message="conv1: ", summarize=10)
            lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=is_train), 0.2)

            # 2nd hidden layer
            conv2 = tf.layers.conv2d_transpose(lrelu1, 64, [4, 4], strides=(2, 2), padding='valid', kernel_initializer=self.kernel_initializer)
            # conv2 = tf.Print(conv2, [tf.shape(conv2)], message="conv2: ", summarize=10)
            lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=is_train), 0.2)

            # 3rd hidden layer
            conv3 = tf.layers.conv2d_transpose(lrelu2, 64, [4, 4], strides=(2, 2), padding='same', kernel_initializer=self.kernel_initializer)
            # conv3 = tf.Print(conv3, [tf.shape(conv3)], message="conv3: ", summarize=10)
            lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=is_train), 0.2)

            # 3rd hidden layer
            conv4 = tf.layers.conv2d_transpose(lrelu3, 32, [4, 4], strides=(2, 2), padding='valid',
                                               kernel_initializer=self.kernel_initializer)
            # conv4 = tf.Print(conv4, [tf.shape(conv4)], message="conv4: ", summarize=10)
            lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=is_train), 0.2)

            # Logits for Bernoulli distribution
            bernoulli_logits = tf.layers.conv2d_transpose(lrelu4, 1, kernel_size=3, strides=1, padding='same', kernel_initializer=self.kernel_initializer)

        return bernoulli_logits

    def _sample_z(self, mean, sigma, shape=None):
        if shape is None: shape = tf.shape(mean)
        return tf.random_normal(shape=shape, mean=mean, stddev=sigma)

    def lower_bound(self, x):
        print("lower bound")
        # Encoder: both mean and variance have sodimension (n_samples, z_dim)
        z_mean, z_sigma = self.encode(x)

        z_var = tf.square(z_sigma)
        z_log_var = tf.log(z_var)

        # Sample from the Gaussian distribution parametrized by mean,sigma
        z_sample = self._sample_z(z_mean, z_sigma)

        # Compute the KL divergence
        kl_divergence = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - z_var, axis=1)

        # Decoder
        p_x_given_z_logits = self.decode(z_sample)
        log_prop_data = tf.reduce_sum(tf.distributions.Bernoulli(
            logits=p_x_given_z_logits).log_prob(x), axis=[1, 2, 3])

        # Compute the lower bound
        lower_bound = -kl_divergence + log_prop_data

        return lower_bound

    def mean_x_given_z(self, z):
        # Returns tensor containing the mean of p(X|Z=z) for each of the given points
        return tf.sigmoid(self.decode(z))  # (n_samples, n_dim)

    def sample(self, n_samples):
        # Generate N samples from your model
        z_samples = self._sample_z(mean=0.0, sigma=1.0, shape=(n_samples, self.z_dim))
        return tf.distributions.Bernoulli(probs=self.mean_x_given_z(z_samples)).sample()

    def test_coder(self, x):
        # Encoder: both mean and variance have sodimension (n_samples, z_dim)
        z_mean, z_sigma = self.encode(x)
        z_sample = self._sample_z(z_mean, z_sigma)
        p_x_given_z_logits = self.decode(z_sample)

        return  x, p_x_given_z_logits


# # num_images = 51200
# # (x_size, y_size) = (30, 30)
# # data = np.empty((num_images, y_size, x_size, 1), int)
# # data_y = np.empty((num_images,), int)
# # f_strings = []
#
#
# # def load_images(directory, show_load=False):
# #     count = 0
# #     for subdir, dirs, files in os.walk(directory):
# #         for file in files:
# #             # print os.path.join(subdir, file)
# #             # print(file)
# #             if file.endswith(".png"):
# #                 temp_data = cv2.imread(os.path.join(subdir, file), 0)
# #                 # data[count] = np.expand_dims(cv2.resize(temp_data, (28, 28)), -1)
# #                 data[count] = np.expand_dims(temp_data, -1)
# #                 # f_strings.append(os.path.join(subdir, file))
# #                 data_y[count] = int(file.split("_")[2])
# #                 count += 1
# #             if show_load and count % (num_images / 10) == 0:
# #                 print(str(count / (num_images / 10) * 10) + "%")
# #     return
#
# # show samples
# # def show_samples(sess, vae):
# #     im_samples = sess.run(vae.sample(100))
# #     for i in range(100):
# #         cv2.imshow("Example", np.squeeze(im_samples[i]*255, axis=-1).astype(np.uint8))
# #         if cv2.waitKey(0) == ord('q'):
# #             break
# #
# #
# # load_images("generated_images/")
# #
# # # parameters
# # train = True
# # load_model = False
# # save_model = True
# #
# # # save/load parameters
# # save_path = "/tmp/model_100.ckpt"
# #
# # # tensorboard parameters
# # board = True
# # board_map = "board"
# # loss_name = 'loss_100_50'
# #
# # # train parameters
# # epoch = 50
# # batch_size = 512
# # alpha_start = 100
# # lower_alpha = False
# # alpha = alpha_start
# # learning_rate = tf.placeholder(tf.float32, shape=[])
# #
# # # setup data for training
# # total_iter = epoch * (num_images / batch_size)
# # dataset = tf.data.Dataset.from_tensor_slices((data, data_y))
# # dataset.shuffle(num_images)
# # dataset = dataset.repeat(epoch)
# # batch = dataset.batch(batch_size)
# # iter = batch.make_one_shot_iterator()
# #
# # # setup model
# # vae = VariationalAutoencoder((x_size, y_size), 15)
# # (x, y) = iter.get_next()
# # x = tf.to_float(x)
# # lower_bound = vae.lower_bound(x)
# # loss = -tf.reduce_mean(lower_bound)
# # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
# # init = tf.global_variables_initializer()
# #
# # # tensorboard
# # if board:
# #     tf.summary.scalar(loss_name, loss)
# #     merged = tf.summary.merge_all()
# #
# # # saver to for trained model
# # saver = tf.train.Saver()
# #
# # # start session
# # sess = tf.Session()
# #
# # # load model
# # if load_model:
# #     saver.restore(sess, save_path)
# #     print("Model restored.")
# #
# #
# # # train
# # if train:
# #     # tensorboard
# #     if board:
# #         writer = tf.summary.FileWriter(board_map, sess.graph)
# #
# #     sess.run(init)
# #     time_start = time.time()
# #     for i in range(total_iter):
# #         summary, _ = sess.run([merged, optimizer], feed_dict={learning_rate: alpha})
# #         writer.add_summary(summary, i)
# #
# #         # print percentage done + estimated time to arrival
# #         if i % (total_iter/100) is 0:
# #             eta_min = int((time.time()-time_start)/(float(i+1)/total_iter)-(time.time()-time_start))/60
# #             print(str(100*i/total_iter)+ "%   eta: " + str(eta_min/60)+"H"+str(eta_min%60)+"M")
# #
# #         # lower alpha mid training if lower_alpha is True
# #         if lower_alpha:
# #             if i == total_iter/100:
# #                 alpha = alpha/10.0
# #                 print("update alpha1: " + str(alpha))
# #             elif i == total_iter/20:
# #                 alpha = alpha/10.0
# #                 print("update alpha2: " + str(alpha))
# #             elif i == total_iter/5:
# #                 alpha = alpha/10.0
# #                 print("update alpha3: " + str(alpha))
# #             elif i == total_iter/2:
# #                 alpha = alpha/10.0
# #                 print("update alpha4: " + str(alpha))
# #
# #     # save model after training
# #     if save_model:
# #         save_path = saver.save(sess, save_path)
# #         print("Model saved in path: %s" % save_path)
# #
# #
# # # show some generated samples
# # show_samples(sess, vae)
#
# def load_images(binarize=True):
#     count = 0
#     surfaces = [i * 9 for i in [32, 64, 96, 128, 160, 192, 224, 256]]
#     object_nums = range(0, 21)
#     im_per_class = 10
#     num_images = len(surfaces) * len(object_nums) * im_per_class
#     (x_size, y_size) = (90, 90)
#     x = np.empty((num_images, y_size, x_size, 1), int)
#     y = np.empty((num_images,), int)
#     for sum_surface in surfaces:
#         for num_obj in object_nums:
#             directory = "/home/dico/Documents/generated_images_100/surf" + str(sum_surface) + "_obj"+ str(num_obj)
#             for i in range(im_per_class):
#                 temp_data = cv2.imread(directory + "/image_" + str(sum_surface) + "_" + str(num_obj) + "_" + str(i) + ".png", 0)
#                 x[count] = np.expand_dims(temp_data, -1)
#                 y[count] = num_obj
#                 count += 1
#     np.random.shuffle(x)
#     x_train, x_test = x[:int(num_images*9/10),:].astype(np.float32), x[int(num_images*9/10):,:].astype(np.float32)
#     if binarize:
#         x_train = (x_train>0.5).astype(x_test.dtype)
#         x_test  = (x_test>0.5).astype(x_test.dtype)
#     return x_train, x_test
#
# # def load_mnist_images(binarize=True):
# #     from tensorflow.examples.tutorials.mnist import input_data
# #     mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
# #     x_train = mnist.train.images.reshape(-1, 28, 28, 1)
# #     x_test  = mnist.test.images.reshape(-1, 28, 28, 1)
# #     if binarize:
# #         x_train = (x_train>0.5).astype(x_train.dtype)
# #         x_test  = (x_test>0.5).astype(x_test.dtype)
# #     return x_train, x_test
#
# # show samples
# def show_samples(sess, vae):
#     im_samples = sess.run(vae.sample(100))
#     for i in range(100):
#         cv2.imshow("Example", np.squeeze(im_samples[i]*255, axis=-1).astype(np.uint8))
#         if cv2.waitKey(0) == ord('q'):
#             break
#
#
# def train_vae(config):
#     # save_path = "/models/model30.ckpt"
#     save_path = config.model_path + "model" + config.model_name + ".ckpt"
#     load_model = False
#     save_model = False
#
#     # Load dataset
#     x_train, x_test = load_images(binarize=True)
#     n_samples, im_height, im_width, _ = x_train.shape
#
#     # Data pipeline
#     data_train = tf.data.Dataset.from_tensor_slices(x_train)
#     data_train = data_train.shuffle(buffer_size=10000)
#     data_train = data_train.repeat(config.num_epochs)
#     data_train = data_train.batch(config.batch_size)
#
#     data_test = tf.data.Dataset.from_tensor_slices(x_test).batch(config.batch_size)
#     iterator = tf.data.Iterator.from_structure(data_train.output_types, data_train.output_shapes)
#     im_batch = iterator.get_next()
#     init_train_data_op = iterator.make_initializer(data_train)
#     init_test_data_op  = iterator.make_initializer(data_test)
#
#     # Build VAE model
#     model = VariationalAutoencoder(
#         x_shape=(im_height, im_width),
#         z_dim=config.z_dim,
#     )
#
#     global_step = tf.train.get_or_create_global_step()
#
#     # Lower bound and loss
#     print("x_minibatch", im_batch.shape)
#     code = model.encode(im_batch)
#
#     # lower_bound = model.lower_bound(im_batch)
#     # loss = -tf.reduce_mean(lower_bound)
#     #
#     # ema = tf.train.ExponentialMovingAverage(decay=0.99, zero_debias=True)
#     # main_averages_op = ema.apply([loss])
#     #
#     # # Optimizer
#     # optimizer = tf.train.AdamOptimizer(config.learning_rate)
#     # apply_gradients_op = optimizer.minimize(loss, global_step)
#     # with tf.control_dependencies([apply_gradients_op]):
#     #     train_op = tf.group(main_averages_op)
#     #
#     # # Build the code for plotting the manifold.
#     # z0, z1 = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
#     # z_grid_points = tf.constant(np.concatenate([z0.flatten()[:, None], z1.flatten()[:, None], np.zeros((5**2, config.z_dim-2))], axis=1).astype(np.float32))
#     # x_points = model.mean_x_given_z(z_grid_points)
#     # x_points = tf.reshape(x_points, shape=[-1, 90, 90, 1])
#     #
#     # # Sample from the model, for visualization
#     # samples = model.sample(10)
#     # samples = tf.reshape(samples, [-1,90,90,1])
#     # samples = tf.cast(samples, tf.float32)
#     #
#     # # Placeholder for test loss
#     # pl_test_loss = tf.placeholder(tf.float32, [], name="test_loss")
#     #
#     # # Initialize summaries
#     # train_scalar_summaries_op = tf.summary.merge([tf.summary.scalar("train/loss", loss)])
#     # test_scalar_summaries_op  = tf.summary.merge([tf.summary.scalar("test/loss", pl_test_loss)])
#     #
#     # image_summaries = [
#     #     tf.summary.image("mean_x_given_z", x_points, max_outputs=30),
#     #     tf.summary.image("samples", samples, max_outputs=30)
#     # ]
#     # image_summaries_op = tf.summary.merge(image_summaries)
#     # summary_writer = tf.summary.FileWriter(config.summary_path+"run_{}".format(datetime.now().strftime("%Y%m%d_%H%M")))
#     #
#     # # Initialize TensorFlow session
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_mem_frac)
#     saver = tf.train.Saver()
#     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=config.log_device_placement))
#     #
#     # # load model
#     # if load_model:
#     #     saver.restore(sess, save_path)
#     #     print("Model restored.")
#     #
#     sess.run(init_train_data_op) # Initialize the variables of the data-loader.
#     sess.run(tf.global_variables_initializer())  # Initialize the model parameters.
#     #
#     # num_steps = int((config.num_epochs * n_samples) / config.batch_size)
#     # examples_per_second = 0
#     #
#     # train_ops = {
#     #     'train_op': train_op,
#     #     'train_lower_bound': tf.reduce_mean(lower_bound),
#     #     'train_loss': ema.average(loss),
#     #     'train_summary': train_scalar_summaries_op
#     # }
#
#     sess.run(code)
#     sess.run(lower_bound)
#     if save_model:
#         save_path = saver.save(sess, save_path)
#         print("Model saved in path: %s" % save_path)
#     # show some generated samples
#     # show_samples(sess, model)
#
#
# if __name__ == '__main__':
#     print("training...")
#
#     # Parse training configuration
#     parser = argparse.ArgumentParser()
#
#     # Model params
#     parser.add_argument('--z_dim', type=int, default=20, help='Dimensionality of z')#defaul=2
#
#     # Training params
#     parser.add_argument('--batch_size', type=int, default=256, help='Number of examples to process in a batch')#default=100
#     parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')#default=0.003
#     parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')#default=1000
#
#     # Print, sampling and testing frequency
#     parser.add_argument('--print_every', type=int, default=25, help='Frequency of printing model train performance')
#     parser.add_argument('--sample_every', type=int, default=1000, help='Frequency of sampling from model')#default=500
#     parser.add_argument('--test_every', type=int, default=200, help='Frequency of testing model performance')
#
#     # Misc params
#     parser.add_argument('--gpu_mem_frac', type=float, default=0.5, help='Fraction of GPU memory to allocate')#default=0.5
#     parser.add_argument('--log_device_placement', type=bool, default=False, help='Log device placement for debugging')
#     parser.add_argument('--summary_path', type=str, default="./logs/", help='Output path for summaries')#default="./summaries/"
#     parser.add_argument('--model_path', type=str, default="./models/", help='Output path for the model')
#     parser.add_argument('--model_name', type=str, default="100", help='Output name for the model')
#
#     config = parser.parse_args()
#
# train_vae(config)
