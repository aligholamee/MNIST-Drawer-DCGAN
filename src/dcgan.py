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
import numpy as np

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
        
        # CNN_T Layer 1
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=2, padding='same', activation=tf.nn.leaky_relu, name="CNN_4")
        # Dropout 2
        x = tf.layers.dropout(x, keep_prob, name="D_5")
        # Batch_norm 2
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)

        # CNN_T Layer 2
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=2, padding='same', activation=tf.nn.leaky_relu, name="CNN_5")
        # Dropout 3
        x = tf.layers.dropout(x, keep_prob, name="D_6")
        # Batch_norm 3
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)

        # CNN_T Layer 3
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=1, padding='same', activation=tf.nn.leaky_relu, name="CNN_6")
        x = tf.layers.dropout(x, keep_prob)

        # Batch_norm 4
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)

        # CNN_T Layer 4
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=1, strides=1, padding='same', activation=tf.nn.sigmoid, name="CNN_7")
    
        return x
        
G = generator(NOISE_INPUT, KEEP_PROB, IS_TRAINING)
D_REAL = discriminator(X_INPUT)
D_FAKE = discriminator(G, reuse=True)

VARS_G = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
VARS_D = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]


D_REG = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), VARS_D)
G_REG = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), VARS_G)

LOSS_D_REAL = binary_cross_entropy(tf.ones_like(D_REAL), D_REAL)
LOSS_D_FAKE = binary_cross_entropy(tf.zeros_like(D_FAKE), D_FAKE)
LOSS_G = tf.reduce_mean(binary_cross_entropy(tf.ones_like(D_FAKE), D_FAKE))
LOSS_D = tf.reduce_mean(0.5 * (LOSS_D_REAL + LOSS_D_FAKE))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer_d = tf.train.RMSPropOptimizer(learning_rate=0.00015).minimize(LOSS_D + D_REG, var_list=VARS_D)
    optimizer_g = tf.train.RMSPropOptimizer(learning_rate=0.00015).minimize(LOSS_G + G_REG, var_list=VARS_G)
    
    
SESS = tf.Session()
SESS.run(tf.global_variables_initializer())

for i in range(60000):
    train_d = True
    train_g = True
    keep_prob_train = 0.6 # 0.5
    
    
    n = np.random.uniform(0.0, 1.0, [BATCH_SIZE, N_NOISE]).astype(np.float32)   
    batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=BATCH_SIZE)[0]]  
    
    d_real_ls, d_fake_ls, g_ls, d_ls = SESS.run([LOSS_D_REAL, LOSS_D_FAKE, LOSS_G, LOSS_D], feed_dict={X_INPUT: batch, NOISE_INPUT: n, KEEP_PROB: keep_prob_train, IS_TRAINING:True})
    
    d_real_ls = np.mean(d_real_ls)
    d_fake_ls = np.mean(d_fake_ls)
    g_ls = g_ls
    d_ls = d_ls
    
    if g_ls * 1.5 < d_ls:
        train_g = False
        pass
    if d_ls * 2 < g_ls:
        train_d = False
        pass
    
    if train_d:
        SESS.run(optimizer_d, feed_dict={NOISE_INPUT: n, X_INPUT: batch, KEEP_PROB: keep_prob_train, IS_TRAINING:True})
        
        
    if train_g:
        SESS.run(optimizer_g, feed_dict={NOISE_INPUT: n, KEEP_PROB: keep_prob_train, IS_TRAINING:True})