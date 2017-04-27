# Lab 11 MNIST and Deep learning CNN
import tensorflow as tf
# import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

# hyper parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 100


class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.training = tf.placeholder(tf.bool)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 784])

            # img 28x28x1 (black/white), Input Layer
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout1 = tf.layers.dropout(inputs=pool1,
                                         rate=0.7, training=self.training)

            # Convolutional Layer #2 and Pooling Layer #2
            #conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3],
            #                         padding="SAME", activation=tf.nn.relu)
            # https://www.tensorflow.org/api_docs/python/tf/nn/separable_conv2d

            self.W2_depthwise_filter = tf.get_variable(shape=[3, 3, dropout1.get_shape().as_list()[3], 1],
                                                       initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                       name='W2_depthwise_weight')
            self.W2_pointwise_weight = tf.get_variable(shape=[1, 1,  dropout1.get_shape().as_list()[3] * 1, 64],
                                                    initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                    name='W2_pointwise_weight')
            conv2 = tf.nn.separable_conv2d(input=dropout1,
                                           depthwise_filter=self.W2_depthwise_filter,
                                           pointwise_filter=self.W2_pointwise_weight,
                                           strides=[1, 1, 1, 1],
                                           padding='SAME')
            #conv2 = tf.nn.relu(conv2)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2,
                                         rate=0.7, training=self.training)

            # Convolutional Layer #2 and Pooling Layer #2
            #conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3],
            #                         padding="same", activation=tf.nn.relu)

            self.W3_depthwise_filter = tf.get_variable(shape=[3, 3, dropout2.get_shape().as_list()[3], 1],
                                                       initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                       name='W3_depthwise_weight')
            self.W3_pointwise_weight = tf.get_variable(shape=[1, 1, dropout2.get_shape().as_list()[3] * 1, 128],
                                                       initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                       name='W3_pointwise_weight')
            conv3 = tf.nn.separable_conv2d(input=dropout2,
                                           depthwise_filter=self.W3_depthwise_filter,
                                           pointwise_filter=self.W3_pointwise_weight,
                                           strides=[1, 1, 1, 1],
                                           padding='SAME')
            # conv3 = tf.nn.relu(conv3)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
                                            padding="same", strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3,
                                         rate=0.7, training=self.training)

            # Dense Layer with Relu
            flat = tf.reshape(dropout3, [-1, 128 * 4 * 4])
            dense4 = tf.layers.dense(inputs=flat,
                                     units=625, activation=tf.nn.relu)
            dropout4 = tf.layers.dropout(inputs=dense4,
                                         rate=0.5, training=self.training)

            # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
            self.logits = tf.layers.dense(inputs=dropout4, units=10)

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def evaluate(self, X_sample, y_sample, training=False):
        """Run a minibatch accuracy op"""

        N = X_sample.shape[0]
        correct_sample = 0

        for i in range(0, N, batch_size):
            X_batch = X_sample[i: i + batch_size]
            y_batch = y_sample[i: i + batch_size]
            N_batch = X_batch.shape[0]

            feed = {
                self.X: X_batch,
                self.Y: y_batch,
                self.training: training
            }

            correct_sample += self.sess.run(self.accuracy, feed_dict=feed) * N_batch

        return correct_sample / N

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})

# initialize
sess = tf.Session()
m1 = Model(sess, "m1")

sess.run(tf.global_variables_initializer())

print('Learning Started!')

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
#print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))



print("\nAccuracy Evaluates")
print("-------------------------------")
print('Train Accuracy:', m1.evaluate(mnist.train.images, mnist.train.labels))
print('Test Accuracy:', m1.evaluate(mnist.test.images, mnist.test.labels))