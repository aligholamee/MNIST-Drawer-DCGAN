# ========================================
# [] File Name : dcgan.py
#
# [] Creation Date : January 2018
#
# [] Created By : Ali Gholami (aligholami7596@gmail.com)
# ========================================

"""
    Implementation of a deep convolutional generative adversarial network to portray human faces
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Load the dataset
mnist = input_data.read_data_sets('MNIST_data')

# Reset the graph
tf.reset_default_graph()

# Essential constants
BATCH_SIZE = 64     # TRAINING BATCH SIZE
N_NOISE = 64        # NUMBER OF NOISE INPUTS

# Placeholders definitions
X_INPUT = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name="X_INPUT")
NOISE_INPUT = tf.placeholder(dtype=tf.float32, shape=[None, N_NOISE], name="NOISE_INPUT")
KEEP_PROB = tf.placeholder(dtype=tf.float32, name="KEEP_PROB")      # Used by the dropout layersq0
IS_TRAINING = tf.placeholder(dtype=tf.bool, name="IS_TRAINING")

# Binary cross entropy calculator
def binary_cross_entropy(x, z):
    eps = 1e - 12   # A very small number as epsilon
    return (-(x*tf.log(z+eps) + (1. - x)*tf.log(1. - z + eps)))
    
# Discriminator computational graph generator
def discriminator(input_img, reuse=None, keep_prob=KEEP_PROB):
    
    # The variables name scope
    with tf.variable_scope("discriminator", reuse=reuse):

        # Reshape the input data (Storing in the computational graph)
        cg = tf.reshape(input_img, shape=[-1, 28, 28, 1])

        # First CNN Layer + Dropout
        cg = tf.nn.conv2d(cg, filters=64, kernel_size=5, strides=2, padding='same', activation=tf.nn.leaky_relu, use_cudnn_on_gpu=True, name="CNN_1")
        cg = tf.nn.dropout(cg, keep_prob, name="D_1")

        # Second CNN Layer + Dropout
        cg = tf.nn.conv2d(cg, filters=64, kernel_size=5, strides=2, padding='same', activation=tf.nn.leaky_relu, use_cudnn_on_gpu=True, name="CNN_2")
        cg = tf.nn.dropout(cg, keep_prob, name="D_2")

        # Third CNN Layer + Dropout
        cg = tf.nn.conv2d(cg, filters=64, kernel_size=5, strides=1, padding='same', activation=tf.nn.leaky_relu, use_cudnn_on_gpu=True, name="CNN_3")
        cg = tf.nn.dropout(cg, keep_prob, name="D_3")

        # Flatten the cg 
        cg = tf.contrib.layers.flatten(cg)

        # Wrap everything in a densly-connected layer
        cg = tf.layers.dense(cg, units=128, activation=tf.nn.leaky_relu)
        cg = tf.layers.dense(cg, units=1, activation=tf.nn.sigmoid)

        # Return the computational graph for the discriminator
        return cg





