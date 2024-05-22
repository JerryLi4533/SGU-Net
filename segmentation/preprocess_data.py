
import os
import numpy as np
import cv2
import shutil

def make_dirs(dir_path_list):
    for dir_name in dir_path_list:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            os.makedirs(dir_name)
        else:
            os.makedirs(dir_name)

def crop_training_images(raw_data_dir,raw_labels_dir,patch_size,new_path_list,crop_flag):
    raw_images_list = os.listdir(raw_data_dir)
    image_patch_id = 0
    train_images_dir = new_path_list[0]
    train_labels_dir = new_path_list[1]
    val_images_dir = new_path_list[2]
    val_labels_dir = new_path_list[3]

    #-------------iterate all raw data-----------------
    for i in range(len(raw_images_list)):
        raw_image_name = raw_images_list[i]

        raw_label_name = raw_image_name
        raw_image = cv2.imread(os.path.join(raw_data_dir,raw_image_name))
        raw_label = cv2.imread(os.path.join(raw_labels_dir,raw_label_name))
        if crop_flag ==0:
            raw_image = cv2.resize(raw_image, (patch_size,patch_size))
            raw_label = cv2.resize(raw_label, (patch_size, patch_size))
        H, W = raw_image.shape[0], raw_image.shape[1]
        crop_interval = int(patch_size*0.8)
        x_mod = (W - patch_size) // crop_interval
        y_mod = (H - patch_size) // crop_interval
        x_len = x_mod + 1 if (W - patch_size) % crop_interval == 0 else x_mod + 2
        y_len = y_mod + 1 if (H - patch_size) % crop_interval == 0 else y_mod + 2
        # -------------crop one image-----------------
        for xi in range(x_len):
            for yi in range(y_len):
                patch_image_name = 'image_' + str(image_patch_id) + '.jpg'
                patch_label_name = 'image_' + str(image_patch_id) + '.jpg'
                # patch_label_name = patch_image_name + '.png'
                rand_data = np.random.rand()
                if rand_data>0.1:
                    patch_image_savedir = os.path.join(train_images_dir, patch_image_name)
                    patch_label_savedir = os.path.join(train_labels_dir, patch_label_name)
                else:
                    patch_image_savedir = os.path.join(val_images_dir, patch_image_name)
                    patch_label_savedir = os.path.join(val_labels_dir, patch_label_name)

                point_x0 = xi * crop_interval
                point_y0 = yi * crop_interval
                if yi == y_mod + 1:
                    point_y0 = H - patch_size
                if xi == x_mod + 1:
                    point_x0 = W - patch_size
                image_patch = raw_image[point_y0:(point_y0 + patch_size), point_x0:(point_x0 + patch_size), :].copy()
                image_label_patch = raw_label[point_y0:(point_y0 + patch_size), point_x0:(point_x0 + patch_size), :].copy()
                # TODO assert if image_label_pathch contains objects
                nozero_num = np.where(image_label_patch==0)
                if len(nozero_num[0])>0:
                    cv2.imencode('.jpg', image_patch)[1].tofile(patch_image_savedir)
                    image_label_patch = cv2.cvtColor(image_label_patch, cv2.COLOR_BGR2GRAY)
                    cv2.imencode('.jpg', image_label_patch)[1].tofile(patch_label_savedir)
                    print('image_patch_id: ', image_patch_id)
                    image_patch_id += 1



if __name__ == "__main__":
    raw_images_dir = './images'
    raw_labels_dir = './masks'

    new_training_dir = '../datasets/images/train/'
    new_training_label_dir = '../datasets/labels/train/'
    new_val_dir = '../datasets/images/val/'
    new_val_label_dir = '../datasets/labels/val/'
    patch_size = 512
    crop_flag = 0
    dir_path_list = [new_training_dir,new_training_label_dir,new_val_dir,new_val_label_dir]
    make_dirs(dir_path_list)
    #TODO CROP IAMGES
    #
    crop_training_images(raw_images_dir, raw_labels_dir,patch_size,dir_path_list,crop_flag)




