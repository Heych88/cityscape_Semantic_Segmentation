import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import cv2

from sklearn.utils import shuffle
from labels import cityscapes_labels, kitti_labels


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def gen_batch_function(img_data, image_shape, num_classes):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """

        shuf_img = shuffle(img_data)
        for offset in range(0, len(shuf_img), batch_size):
            batch_samples = shuf_img[offset:offset + batch_size]
            images = []
            gt_images = []

            for image_file in batch_samples:

                image = scipy.misc.imread(image_file[0])
                gt_image = scipy.misc.imread(image_file[1])

                # convert the gt_image label to onehot encoding
                #https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
                onehot = (np.arange(num_classes) == gt_image[:, :, None] - 1).astype(int)

                images.append(image)
                gt_images.append(onehot)

            yield shuffle(np.array(images), np.array(gt_images))

    return get_batches_fn

# function for colorizing a label image:
def testDataToColorImg(img, image_shape):

    color_palette = {label.id : label.color for label in cityscapes_labels}  # all the gt_image values observed in the image data

    img_height = image_shape[0]
    img_width = image_shape[1]

    out_image = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            pixel_value = img[row, col]
            out_image[row, col] = np.array(color_palette[pixel_value])

    return out_image

def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape, is_training):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep probability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, '*_leftImg8bit.png')):

        image = cv2.resize(cv2.imread(image_file), (image_shape[1], image_shape[0]))

        logit = sess.run(logits, feed_dict={image_pl: [image], keep_prob: 1, is_training: False})
        y = (tf.nn.softmax(logit)).eval()

        gray = np.reshape(y.argmax(axis=1), image_shape)
        colour = np.uint8(testDataToColorImg(gray, image_shape))
        street = cv2.addWeighted(image, 0.35, colour, 0.65, 0)

        yield os.path.basename(image_file), np.array(street)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, is_training):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print('Training Finished. Saving test images')
    # Run NN on test images and save them to HD
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'cityscapes/leftImg8bit/train/lindau'), image_shape, is_training)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

    print('Images saved to: {}'.format(output_dir))
