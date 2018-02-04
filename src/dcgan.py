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
KEEP_PROB = tf.placeholder(dtype=tf.float32, name="KEEP_PROB")
IS_TRAINING = tf.placeholder(dtype=tf.bool, name="IS_TRAINING")
