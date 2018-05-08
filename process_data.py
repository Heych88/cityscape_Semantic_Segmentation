import cv2
import os
from pathlib import Path
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.utils import shuffle
from labels import cityscapes_labels, kitti_labels

data_dir = r"./data/"  # location of all the data relative to the programs directory
output_dir = data_dir + "processed_data/" # location of the processed data

# get the path to all training images and their corresponding label image:
#train_img_list = []

# create a function mapping id to trainId
# https://stackoverflow.com/questions/13572448/change-values-in-a-numpy-array
def mapLabel(data):
    # palette must be given in sorted order
    #palette = None # all the possible values that are in the data
    #key = None # key gives the new values you wish palette to be mapped to.
    palette = [label.id for label in cityscapes_labels] #[i-1 for i in range(35)] # all the gt_image values observed in the image data
    key = np.array([label.trainId for label in cityscapes_labels]) #[2 for i in range(35)]) # conversion of original gt_image values into training id's

    index = np.digitize(data.ravel(), palette, right=True)

    return data #np.array(key[index].reshape(data.shape))

def translateImage(image, gt_img, file_name, new_file_name, image_shape):
    # Translation of image data to create more training data.
    # The new image names will include the offset pixel count used.
    # img : 3D training image
    # gt_img : 3D training label image
    # file_name: name of the file being processed
    # new_file_name: additional file name to be added to each processed image
    # distance : max distance of transformed images
    # step : steps between transformed images
    # return : list of new images and corresponding label data

    address_list = [] # address list of the new training images

    name_split = file_name.split(".")
    img_file_name = output_dir + name_split[0] + new_file_name + '_bright_.png'
    cv2.imwrite(img_file_name, image)

    #if first_image:
    # only create one ground truth image for every gamma adjustment to save memory
    first_gt_img_file_name = output_dir + 'gt_' + name_split[0] + new_file_name + '_bright_.png'
    new_gt_img = mapLabel(gt_img) # map ground truths to training labels
    cv2.imwrite(first_gt_img_file_name, new_gt_img)

    address_list.append([img_file_name, first_gt_img_file_name])

    return address_list


def loadAllData(train_imgs_dir, train_gt_dir, image_shape, folder_num, tot_folders):
    gt_file_names = os.listdir(train_gt_dir)

    address_list = []

    for file_num, gt_file_name in enumerate(gt_file_names):

        name_split = gt_file_name.split("_gtFine_")

        if(name_split[1] == 'labelIds.png'):
            print("\rprocessing file {}/{} in folder {}/{}     ".format(1 + (file_num)//4, len(gt_file_names)//4,
                                                                   folder_num, tot_folders), end=' ')
            # open and resize the input images
            img_path = train_gt_dir + gt_file_name
            gt_img = cv2.resize(cv2.imread(img_path, -1), (image_shape[1], image_shape[0]))

            file_name = name_split[0] + '_leftImg8bit.png'
            img_path = train_imgs_dir + file_name
            img = cv2.resize(cv2.imread(img_path, -1), (image_shape[1], image_shape[0]))

            # translate the input images into the data folder
            address_list.extend(translateImage(img, gt_img, file_name, "normal", image_shape))
            # translate the input images horizontally and flipped
            address_list.extend(translateImage(cv2.flip(img, 1), gt_img, file_name, "horz_flip", image_shape))

    return address_list


def getData(image_shape):
    """
    Collects all the data and pre-processes them for training.
    Saves the list of processed images for future quick loading.
    Delete 'cityscape_list.p' or 'kitti_list.p' in '/data/processed_data/' if the data is to be re-processed
    :param image_shape: image dimensions to resize all images too
    :param use_cityscape: Which data-set to use, True => cityscape, False => kitti
    :return: list of all the available processed images
    """

    train_imgs_dir = data_dir + "cityscapes/leftImg8bit/train/"
    train_gt_dir = data_dir + "cityscapes/gtFine/train/"
    address_list = []

    # check if the data has already been processed and if so load the pickled data list
    saved_file_dir = output_dir + "cityscape_list.p"
    if Path(saved_file_dir).exists():
        print("Loading data from ", saved_file_dir)
        global train_img_list
        address_list = pickle.load(open(saved_file_dir, "rb"))
    else:
        folder_num = 1
        # the cityscape data set is broken into different folders.
        # Iterate through each and save the data into the one folder
        for root, dirs, files in os.walk(train_gt_dir, topdown=False):
            for name in dirs:
                imgs_dir = (os.path.join(train_imgs_dir, name) + '/')
                gt_dir = (os.path.join(root, name) + '/')
                address_list.extend(loadAllData(imgs_dir, gt_dir, image_shape, folder_num, len(dirs)))
                folder_num += 1

        pickle.dump(address_list, open(saved_file_dir, "wb+"))

    print("\nTotal data ", len(address_list))
    return shuffle(address_list)
