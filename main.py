import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import numpy as np
import model
#import scoring_utils as su
from pathlib import Path

import process_data
from sklearn.model_selection import train_test_split

tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES

validation_generator = None

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=correct_label)
    loss = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return logits, optimizer, loss


def evaluate(logits, labels, num_classes):

    flat_soft = tf.reshape(tf.nn.softmax(logits=logits), [-1, num_classes])
    flat_labels = tf.reshape(labels, [-1, num_classes])

    mean_iou = tf.metrics.mean_iou(flat_labels, flat_soft, num_classes)

    return mean_iou


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, is_training):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    saver = tf.train.Saver()

    for epoch in range(epochs):
        batch = 0
        for images, labels in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: images, correct_label: labels, keep_prob: 1.0, learning_rate:0.0002, is_training: True})
            batch += 1

            if batch % 100 == 0:
                iou = sess.run([val_mean_iou],
                               feed_dict={input_image: images, correct_label: labels, keep_prob: 1.0, is_training: False})
                iou_sum = iou[0][0]

                print('Epoch {}, batch: {}, loss: {}, IoU: {} '.format(epoch + 1, batch, loss, iou_sum))

                # Save the variables to disk.
                saver.save(sess, './model/checkpoints/model_iter', global_step=epoch)
            else:
                print('Epoch {}, batch: {}, loss: {} '.format(epoch + 1, batch, loss))


def run(image_shape, num_classes, train_data, val_data):
    epochs = 15
    batch_size = 2

    data_dir = './data'
    runs_dir = './runs'

    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    label_shape = (None,) + image_shape + (num_classes,)
    correct_label = tf.placeholder(tf.float32, label_shape)
    img_shape = (None,) + image_shape + (3,)
    input_image = tf.placeholder(tf.float32, img_shape, name='input')
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)

    with tf.Session() as sess:
        # Create function to get batches
        train_generator = helper.gen_batch_function(train_data, image_shape, num_classes)
        global validation_generator
        validation_generator = helper.gen_batch_function(val_data, image_shape, num_classes)

        restore_dir = "./model/checkpoints/model_final"
        # Load model
        '''if Path(restore_dir + '.meta').exists():
            print("Loading pretrained network")
            imported_meta = tf.train.import_meta_graph(restore_dir + '.meta')
            imported_meta.restore(sess, tf.train.latest_checkpoint('./model/checkpoints/'))

            graph = tf.get_default_graph()

            all_vars = tf.get_collection('vars')
            for v in all_vars:
                v_ = sess.run(v)
                print(v_)

            #nn_last_layer = graph.get_tensor_by_name("model_output_conv:0")
        #else:'''
        nn_last_layer = model.networkModel(input_image, num_classes, 'model', is_training)

        logits, optimizer, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        global val_mean_iou
        val_mean_iou = evaluate(nn_last_layer, correct_label, num_classes)

        train_nn(sess, epochs, batch_size, train_generator, optimizer, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate, is_training)

        # save the trained model
        saver = tf.train.Saver()
        saver.save(sess, restore_dir)
        print("Model saved")

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, is_training)

if __name__ == '__main__':

    image_shape = (256, 512)

    num_classes = 33

    # load and pre-process the training data
    print('Collecting Data')
    train_img_list = process_data.getData(image_shape)
    train_data, val_data = train_test_split(train_img_list, test_size=0.0)
    print("Training data: {}, validation data: {}".format(len(train_data), len(val_data)))
    print('Finished collecting Data')

    # train the model
    run(image_shape, num_classes, train_data, val_data)
