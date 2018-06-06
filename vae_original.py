import numpy as np
import tensorflow as tf


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
            conv1 = tf.layers.conv2d(x, 64, [4, 4], strides=(2, 2), padding='same')
            lrelu1 = lrelu(conv1, 0.2)

            # 2nd hidden layer
            conv2 = tf.layers.conv2d(lrelu1, 128, [4, 4], strides=(2, 2), padding='same', kernel_initializer=self.kernel_initializer)
            lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=is_train), 0.2)

            # 3rd hidden layer
            conv3 = tf.layers.conv2d(lrelu2, 128, [4, 4], strides=(2, 2), padding='same', kernel_initializer=self.kernel_initializer)
            lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=is_train), 0.2)

            # 4th hidden layer
            conv4 = tf.layers.conv2d(lrelu3, 256, [4, 4], strides=(2, 2), padding='same', kernel_initializer=self.kernel_initializer)
            lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=is_train), 0.2)
            lrelu4_flat = tf.layers.flatten(lrelu4)

            # Intermediate dense layer
            dense = tf.layers.dense(lrelu4_flat, 200, activation=tf.nn.relu, kernel_initializer=self.kernel_initializer)

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

            dense = tf.layers.dense(z, units=200, kernel_initializer=self.kernel_initializer, activation=tf.nn.relu)
            decoder_expand = tf.layers.dense(dense, units=(128*4*4), kernel_initializer=self.kernel_initializer, activation=tf.nn.relu)
            decoder_expand = tf.reshape(decoder_expand, [-1,4,4,128])

            # 1st hidden layer
            conv1 = tf.layers.conv2d_transpose(decoder_expand, 128, [4, 4], strides=(1, 1), padding='valid', kernel_initializer=self.kernel_initializer)
            lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=is_train), 0.2)

            # 2nd hidden layer
            conv2 = tf.layers.conv2d_transpose(lrelu1, 64, [4, 4], strides=(2, 2), padding='same', kernel_initializer=self.kernel_initializer)
            lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=is_train), 0.2)

            # 3rd hidden layer
            conv3 = tf.layers.conv2d_transpose(lrelu2, 64, [4, 4], strides=(2, 2), padding='valid', kernel_initializer=self.kernel_initializer)
            lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=is_train), 0.2)

            # Logits for Bernoulli distribution
            bernoulli_logits = tf.layers.conv2d_transpose(lrelu3, 1, kernel_size=3, strides=1, padding='same', kernel_initializer=self.kernel_initializer)

        return bernoulli_logits

    def _sample_z(self, mean, sigma, shape=None):
        if shape is None: shape = tf.shape(mean)
        return tf.random_normal(shape=shape, mean=mean, stddev=sigma)

    def lower_bound(self, x):

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
            logits=p_x_given_z_logits).log_prob(x), axis=[1,2,3])

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