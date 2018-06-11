import numpy as np
import tensorflow as tf
import time

import matplotlib.pyplot as plt
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
            # x = tf.Print(x, [tf.shape(x)], message="x: ", summarize=10)
            conv1 = tf.layers.conv2d(x, 32, [4, 4], strides=(2, 2), padding='same')
            # conv1 = tf.Print(conv1, [tf.shape(conv1)], message="conv1: ", summarize=10)
            lrelu1 = lrelu(conv1, 0.2)

            # 2nd hidden layer
            conv2 = tf.layers.conv2d(lrelu1, 64, [4, 4], strides=(2, 2), padding='same',
                                     kernel_initializer=self.kernel_initializer)
            # conv2 = tf.Print(conv2, [tf.shape(conv2)], message="conv2: ", summarize=10)
            lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=is_train), 0.2)

            # 3rd hidden layer
            conv3 = tf.layers.conv2d(lrelu2, 128, [4, 4], strides=(2, 2), padding='same',
                                     kernel_initializer=self.kernel_initializer)
            # conv3 = tf.Print(conv3, [tf.shape(conv3)], message="conv3: ", summarize=10)
            lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=is_train), 0.2)

            # 4th hidden layer
            conv4 = tf.layers.conv2d(lrelu3, 256, [4, 4], strides=(2, 2), padding='same',
                                     kernel_initializer=self.kernel_initializer)
            # conv4 = tf.Print(conv4, [tf.shape(conv4)], message="conv4: ", summarize=10)
            lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=is_train), 0.2)
            lrelu4_flat = tf.layers.flatten(lrelu4)


            # Intermediate dense layer
            dense = tf.layers.dense(lrelu4_flat, 100, activation=tf.nn.relu, kernel_initializer=self.kernel_initializer)
            # dense = tf.Print(dense, [tf.shape(dense)], message="dense: ", summarize=10)


            # Parameters of Gaussian distribution
            z_gaussian_params = tf.layers.dense(dense, units=2 * self.z_dim, kernel_initializer=self.kernel_initializer)

            # Mean parameter of Gaussian is unconstrained
            z_mean = z_gaussian_params[:, 0:self.z_dim]

            # Standard deviation of Gaussian must be positive. Therefore, we
            # use a softplus and also add a small epsilon for numerical stability.
            z_sigma = 1e-6 + tf.nn.softplus(z_gaussian_params[:, self.z_dim:])
            # z_mean = tf.Print(z_mean, [tf.shape(z_mean)], message="z_mean: ", summarize=10)
            # z_sigma = tf.Print(z_sigma, [tf.shape(z_sigma)], message="z_sigma: ", summarize=10)

        return z_mean, z_sigma

    def decode(self, z, is_train=True):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            dense = tf.layers.dense(z, units=100, kernel_initializer=self.kernel_initializer, activation=tf.nn.relu)
            decoder_expand = tf.layers.dense(dense, units=(128 * 2 * 2), kernel_initializer=self.kernel_initializer,
                                             activation=tf.nn.relu)
            decoder_expand = tf.reshape(decoder_expand, [-1, 2, 2, 128])
            # decoder_expand = tf.Print(decoder_expand, [tf.shape(decoder_expand)], message="decoder_expand: ", summarize=10)

            # 1st hidden layer
            conv1 = tf.layers.conv2d_transpose(decoder_expand, 128, [4, 4], strides=(2, 2), padding='same',
                                               kernel_initializer=self.kernel_initializer)
            # conv1 = tf.Print(conv1, [tf.shape(conv1)], message="conv1: ", summarize=10)
            lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=is_train), 0.2)

            # 2nd hidden layer
            conv2 = tf.layers.conv2d_transpose(lrelu1, 64, [4, 4], strides=(1, 1), padding='valid',
                                               kernel_initializer=self.kernel_initializer)
            # conv2 = tf.Print(conv2, [tf.shape(conv2)], message="conv2: ", summarize=10)
            lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=is_train), 0.2)

            # 3rd hidden layer
            conv3 = tf.layers.conv2d_transpose(lrelu2, 64, [4, 4], strides=(2, 2), padding='same',
                                               kernel_initializer=self.kernel_initializer)
            # conv3 = tf.Print(conv3, [tf.shape(conv3)], message="conv3: ", summarize=10)
            lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=is_train), 0.2)

            conv4 = tf.layers.conv2d_transpose(lrelu3, 32, [4, 4], strides=(2, 2), padding='valid',
                                               kernel_initializer=self.kernel_initializer)
            # conv4 = tf.Print(conv4, [tf.shape(conv4)], message="conv3: ", summarize=10)
            lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=is_train), 0.2)

            # Logits for Bernoulli distribution
            bernoulli_logits = tf.layers.conv2d_transpose(lrelu4, 1, kernel_size=3, strides=1, padding='same',
                                                          kernel_initializer=self.kernel_initializer)
            # bernoulli_logits = tf.Print(bernoulli_logits, [tf.shape(bernoulli_logits)], message="bernoulli_logits: ", summarize=10)


        return bernoulli_logits

    def _sample_z(self, mean, sigma, shape=None):
        if shape is None: shape = tf.shape(mean)
        return tf.random_normal(shape=shape, mean=mean, stddev=sigma)

    def lower_bound(self, x):
        # x = tf.Print(x, [tf.shape(x)], message="x: ", summarize=10)
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

        return  lower_bound

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

# num_images = 51200
# (x_size, y_size) = (30, 30)
# data = np.empty((num_images, y_size, x_size, 1), int)
# data_y = np.empty((num_images,), int)
# f_strings = []


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

# show samples
# def show_samples(sess, vae):
#     im_samples = sess.run(vae.sample(100))
#     for i in range(100):
#         cv2.imshow("Example", np.squeeze(im_samples[i]*255, axis=-1).astype(np.uint8))
#         if cv2.waitKey(0) == ord('q'):
#             break
#
#
# load_images("generated_images/")
#
# # parameters
# train = True
# load_model = False
# save_model = True
#
# # save/load parameters
# save_path = "/tmp/model_100.ckpt"
#
# # tensorboard parameters
# board = True
# board_map = "board"
# loss_name = 'loss_100_50'
#
# # train parameters
# epoch = 50
# batch_size = 512
# alpha_start = 100
# lower_alpha = False
# alpha = alpha_start
# learning_rate = tf.placeholder(tf.float32, shape=[])
#
# # setup data for training
# total_iter = epoch * (num_images / batch_size)
# dataset = tf.data.Dataset.from_tensor_slices((data, data_y))
# dataset.shuffle(num_images)
# dataset = dataset.repeat(epoch)
# batch = dataset.batch(batch_size)
# iter = batch.make_one_shot_iterator()
#
# # setup model
# vae = VariationalAutoencoder((x_size, y_size), 15)
# (x, y) = iter.get_next()
# x = tf.to_float(x)
# lower_bound = vae.lower_bound(x)
# loss = -tf.reduce_mean(lower_bound)
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
# init = tf.global_variables_initializer()
#
# # tensorboard
# if board:
#     tf.summary.scalar(loss_name, loss)
#     merged = tf.summary.merge_all()
#
# # saver to for trained model
# saver = tf.train.Saver()
#
# # start session
# sess = tf.Session()
#
# # load model
# if load_model:
#     saver.restore(sess, save_path)
#     print("Model restored.")
#
#
# # train
# if train:
#     # tensorboard
#     if board:
#         writer = tf.summary.FileWriter(board_map, sess.graph)
#
#     sess.run(init)
#     time_start = time.time()
#     for i in range(total_iter):
#         summary, _ = sess.run([merged, optimizer], feed_dict={learning_rate: alpha})
#         writer.add_summary(summary, i)
#
#         # print percentage done + estimated time to arrival
#         if i % (total_iter/100) is 0:
#             eta_min = int((time.time()-time_start)/(float(i+1)/total_iter)-(time.time()-time_start))/60
#             print(str(100*i/total_iter)+ "%   eta: " + str(eta_min/60)+"H"+str(eta_min%60)+"M")
#
#         # lower alpha mid training if lower_alpha is True
#         if lower_alpha:
#             if i == total_iter/100:
#                 alpha = alpha/10.0
#                 print("update alpha1: " + str(alpha))
#             elif i == total_iter/20:
#                 alpha = alpha/10.0
#                 print("update alpha2: " + str(alpha))
#             elif i == total_iter/5:
#                 alpha = alpha/10.0
#                 print("update alpha3: " + str(alpha))
#             elif i == total_iter/2:
#                 alpha = alpha/10.0
#                 print("update alpha4: " + str(alpha))
#
#     # save model after training
#     if save_model:
#         save_path = saver.save(sess, save_path)
#         print("Model saved in path: %s" % save_path)
#
#
# # show some generated samples
# show_samples(sess, vae)
