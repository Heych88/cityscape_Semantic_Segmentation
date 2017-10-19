import cv2
import os
import numpy as np

data_dir = "/media/haidyn/Self Driving Car/GIT/my_github/SDC/udacity-sdcnd-Semantic-Segmentation/data/"
output_dir = data_dir + "processed_data/"

# get the path to all training images and their corresponding label image:
train_img_list = []
#train_gt_img_list = []

no_road_val = 76
road_val = 105
other_road_val = 0
image_pad_val = 255

dist = 0

# create a function mapping id to trainId
# https://stackoverflow.com/questions/13572448/change-values-in-a-numpy-array
def mapLabel(data, use_cityscape):
    # palette must be given in sorted order
    palette = None
    key = None
    if(use_cityscape):
        palette = [i-1 for i in range(35)]
        key = np.array([0 for i in range(35)])
        key[8] = 1
    else:
        palette = [other_road_val, no_road_val, road_val, image_pad_val]
        # key gives the new values you wish palette to be mapped to.
        key = np.array([1, 2, 1, 0])
    index = np.digitize(data.ravel(), palette, right=True)

    return np.array(key[index].reshape(data.shape))

def translateImage(img, gt_img, file_name, new_file_name, use_cityscape, distance=0, step=20):
    # Translation of image data to create more training data.
    # The new image names will include the offset pixel count used.
    # img : 3D training image
    # gt_img : 3D training label image
    # file_name: name of the file being processed
    # new_file_name: additional file name to be added to each processed image
    # distance : max distance of transformed images
    # step : steps between transformed images
    # return : list of new images and corresponding label data

    rows, cols, _ = img.shape

    name_split = file_name.split(".")
    img_name = data_dir + "processed_data/" + name_split[0] + '_' + new_file_name +".png"
    cv2.imwrite(img_name, img)


    gt_img_name = data_dir + "processed_data/" + 'gt_' + name_split[0] + '_' + new_file_name + ".png"
    new_gt_img = mapLabel(gt_img, use_cityscape)
    cv2.imwrite(gt_img_name, new_gt_img)
    train_img_list.append([img_name, gt_img_name])

    # add step to include the distance in the image transform
    for offset in range(step, distance+step, step):
        M_right = np.float32([[1, 0, offset], [0, 1, 0]]) # move image right
        M_left = np.float32([[1, 0, -offset], [0, 1, 0]]) # move image left

        # shift the image to the right and append the process image to the list
        img_name = data_dir + "processed_data/" + name_split[0] + '_' + new_file_name + '_right_' + str(offset) + ".png"
        gt_img_name = data_dir + "processed_data/" + 'gt_' + name_split[0] + '_' + new_file_name + '_right_' + str(offset) + ".png"

        new_img = cv2.warpAffine(img, M_right, (cols, rows))
        cv2.imwrite(img_name, new_img)

        new_gt_img = cv2.warpAffine(gt_img, M_right, (cols, rows), borderValue=image_pad_val)
        new_gt_img = mapLabel(new_gt_img, use_cityscape)
        cv2.imwrite(gt_img_name, new_gt_img)
        train_img_list.append([img_name, gt_img_name])

        # shift the image to the left and append the process image to the list
        img_name = data_dir + "processed_data/" + name_split[0] + '_' + new_file_name + '_left_' + str(offset) + ".png"
        gt_img_name = data_dir + "processed_data/" + 'gt_' + name_split[0] + '_' + new_file_name + '_left_' + str(offset) + ".png"

        new_img = cv2.warpAffine(img, M_left, (cols, rows))
        cv2.imwrite(img_name, new_img)

        new_gt_img = cv2.warpAffine(gt_img, M_left, (cols, rows), borderValue=image_pad_val)
        new_gt_img = mapLabel(new_gt_img, use_cityscape)
        cv2.imwrite(gt_img_name, new_gt_img)
        train_img_list.append([img_name, gt_img_name])

def loadAllData(train_imgs_dir, train_gt_dir, image_shape, use_cityscape, folder_num, tot_folders):
    gt_file_names = os.listdir(train_gt_dir)

    for file_num, gt_file_name in enumerate(gt_file_names):

        if(use_cityscape):
            name_split = gt_file_name.split("_gtFine_")

            if(name_split[1] == 'labelIds.png'):
                print("\rprocessing file {}/{} in folder {}/{} ".format(1 + (file_num)//4, len(gt_file_names)//4,
                                                                       folder_num, tot_folders), end=' ')
                # open and resize the input images
                img_path = train_gt_dir + gt_file_name
                gt_img = cv2.resize(cv2.imread(img_path, -1), (image_shape[1], image_shape[0]))

                file_name = name_split[0] + '_leftImg8bit.png'
                img_path = train_imgs_dir + file_name
                img = cv2.resize(cv2.imread(img_path, -1), (image_shape[1], image_shape[0]))

                # translate the input images into the data folder
                translateImage(img, gt_img, file_name, "normal", use_cityscape, distance=dist, step=20)
                # translate the horizontally flipped input images
                translateImage(cv2.flip(img, 1), cv2.flip(gt_img, 1), file_name, "horz_flip", distance=dist, step=20)
                # translate the vertically flipped input images
                translateImage(cv2.flip(img, 0), cv2.flip(gt_img, 0), file_name, "vert_flip", distance=dist, step=20)
                # translate the horizontally and vertically flipped input images
                translateImage(cv2.flip(img, -1), cv2.flip(gt_img, -1), file_name, "vert_horz_flip", distance=dist, step=20)

        else:
            print("\rprocessing file {}/{} in folder {}/{}".format((file_num + 1), len(gt_file_names),
                                                                   folder_num, tot_folders), end=' ')
            # open and resize the input images
            img_path = train_gt_dir + gt_file_name
            gt_img = cv2.resize(cv2.imread(img_path, 0), (image_shape[1], image_shape[0]))

            name_split = gt_file_name.split("_")
            file_name = name_split[0] + '_' + name_split[2]
            img_path = train_imgs_dir + file_name
            img = cv2.resize(cv2.imread(img_path, -1), (image_shape[1], image_shape[0]))

            # translate the input images into the data folder
            translateImage(img, gt_img, file_name, "normal", use_cityscape, distance=dist, step=20)
            # translate the horizontally flipped input images
            translateImage(cv2.flip(img, 1), cv2.flip(gt_img, 1), file_name, "horz_flip", distance=dist, step=20)
            # translate the vertically flipped input images
            translateImage(cv2.flip(img, 0), cv2.flip(gt_img, 0), file_name, "vert_flip", distance=dist, step=20)
            # translate the horizontally and vertically flipped input images
            translateImage(cv2.flip(img, -1), cv2.flip(gt_img, -1), file_name, "vert_horz_flip", distance=dist, step=20)

def getData(image_shape, use_cityscape):

    if(use_cityscape):
        train_imgs_dir = data_dir + "cityscapes/leftImg8bit/train/"
        train_gt_dir = data_dir + "cityscapes/gtFine/train/"

        folder_num = 1

        for root, dirs, files in os.walk(train_gt_dir, topdown=False):
            for name in dirs:
                imgs_dir = (os.path.join(train_imgs_dir, name) + '/')
                gt_dir = (os.path.join(root, name) + '/')
                loadAllData(imgs_dir, gt_dir, image_shape, use_cityscape, folder_num, len(dirs))
                folder_num += 1

    else:
        train_imgs_dir = data_dir + "data_road/training/image_2/"
        train_gt_dir = data_dir + "data_road/training/gt_image_2/"

        loadAllData(train_imgs_dir, train_gt_dir, image_shape, use_cityscape, 1, 1)


    print("\nTotal data size ", len(train_img_list))
    return train_img_list

