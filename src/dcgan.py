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
import sys

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
    eps = sys.float_info.epsilon   # A very small number as epsilon
    return (-(x*tf.log(z+eps) + (1. - x)*tf.log(1. - z + eps)))
    
# Discriminator computational graph generator
def discriminator(input_img, reuse=None, keep_prob=KEEP_PROB):
    
    # The variables name scope
    with tf.variable_scope("discriminator", reuse=reuse):

        # Reshape the input data (Storing in the computational graph)
        cg = tf.reshape(input_img, shape=[-1, 28, 28, 1])

        # First CNN Layer + Dropout
        cg = tf.layers.conv2d(cg, kernel_size=5, filters=64, strides=2, padding='same', activation=tf.nn.leaky_relu, name="CNN_1")
        cg = tf.layers.dropout(cg, keep_prob, name="D_1")

        # Second CNN Layer + Dropout
        cg = tf.layers.conv2d(cg, kernel_size=5, filters=64, strides=2, padding='same', activation=tf.nn.leaky_relu, name="CNN_2")
        cg = tf.layers.dropout(cg, keep_prob, name="D_2")

        # Third CNN Layer + Dropout
        cg = tf.layers.conv2d(cg, kernel_size=5, filters=64, strides=1, padding='same', activation=tf.nn.leaky_relu, name="CNN_3")
        cg = tf.layers.dropout(cg, keep_prob, name="D_3")

        # Flatten the cg 
        cg = tf.contrib.layers.flatten(cg)

        # Wrap everything in a densly-connected layer
        cg = tf.layers.dense(cg, units=128, activation=tf.nn.leaky_relu, name="dense_1")
        cg = tf.layers.dense(cg, units=1, activation=tf.nn.sigmoid, name="dense_2")

        # Return the computational graph for the discriminator
        return cg

# Generator computational graph generator
def generator(latent_z, keep_prob=KEEP_PROB, is_training=IS_TRAINING):
    
    # momentum rate
    momentum = 0.99

    # The variable name scope
    with tf.variable_scope("generator", reuse=None):
        x = latent_z

        WIDTH_HEIGHT = 4     # Width and Height
        NUM_CHANNELS = 1      # Number of image channels

        # Dense Layer 1
        x = tf.layers.dense(x, units=WIDTH_HEIGHT*WIDTH_HEIGHT*NUM_CHANNELS, activation=tf.nn.leaky_relu, name="dense_3")
        # Dropout 1
        x = tf.layers.dropout(x, keep_prob, name="D_4")
        # Batch_norm 1
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)

        # Reshape the generated picture
        x = tf.reshape(x, shape=[-1, WIDTH_HEIGHT, WIDTH_HEIGHT, NUM_CHANNELS])

        # Resize the generated picture
        x = tf.image.resize_images(x, size=[7, 7])
        



