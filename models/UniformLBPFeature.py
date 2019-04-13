# -*- coding: utf-8 -*-
# Li Liu

import os
import numpy as np
from skimage import feature as skif
from skimage import io, transform
import random
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVR

# 全局变量
DATASET_DIR = os.path.join(os.path.dirname(__file__), '../datasets')  # The directory for datasets
FER_DIR = os.path.join(DATASET_DIR, 'fer2013/img') #The directory for fer2013 dataset
CK_DIR = os.path.join(DATASET_DIR, 'Cohn-Kanade/CroppedImages') #The directory for fer2013 dataset
IMG_TYPE = 'png'  # 图片类型


def resize_image(file_in, file_out, width, height):
    img = io.imread(file_in)
    out = transform.resize(img, (width, height),
                           mode='reflect')  # mode {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}
    io.imsave(file_out, out)

# Load images
def load_images(images_list, width, height):
    data = np.zeros((len(images_list), width, height))  # Create array for image data storing
    for index, image in enumerate(images_list):
        image_data = io.imread(image, as_grey=True)
        data[index, :, :] = image_data  # Read images to the arrays
    return data

#rotate binary number
def rotate(arr,patterns):
    l = arr.shape[0]
    for i in range(0,l):
        rarr = np.roll(arr,i)
        sum = 0
        for j in range(0,l):
            sum += rarr[j]<<j
        patterns.append(sum)

#Get all uniform LBP patterns and assign a bin index to them
def assign_bin(lbp_point=8):
    patterns=[]
    for i in range(1,lbp_point):
        arr=np.zeros(lbp_point)
        arr[0:i]=1
        rotate(arr,patterns)    
    patterns.sort()
    m={}
    m[0]=0
    num=1
    for p in patterns:
        m[p]=num
        num+=1
    m[pow(2^lbp_point)-1]=num
    return map

def replace(k,m):
    if m.get(k) == None:
        return max(list(m.values()))+1
    else:
        return m.get(k)

def get_lbp_with_region_feature(images_data, lbp_radius=1, lbp_point=8, row=7, column=6):
    uniform_map = assign_bin(lbp_point)
    fpf_replace=np.frompyfunc(replace,2,1)

    w = images_data.shape[1]
    h = images_data.shape[2]
    n_images = images_data.shape[0]

    bin_num = (lbp_point - 1) * lbp_point + 3
    hist = np.zeros((n_images,row*column*bin_num))

    for i in np.arange(n_images):

        # extract non-rotation-invariant uniform lbp feature from each image
        lbp = skif.local_binary_pattern(images_data[i], lbp_point, lbp_radius, 'nri_uniform')
        
        lbp=np.asarray(lbp)
        lbp = fpf_replace(lbp, uniform_map)
        lbp.reshape(row,h/row,column,w/column)

        
        lbp=np.swapaxes(lbp,1,2).reshape(row*column,-1)
        r = np.apply_along_axis(np.histogram,1,lbp,bins=np.arrange(bin_num+1))
        hist = np.concatenate(r[:,0])

    return hist


def main():
    if not os.path.exists(RESIZE_POS_IMAGE_DIR):
        os.makedirs(RESIZE_POS_IMAGE_DIR)
    if not os.path.exists(RESIZE_NEG_IMAGE_DIR):
        os.makedirs(RESIZE_NEG_IMAGE_DIR)
    
    pos_file_path_list = map(lambda x: os.path.join(POS_IMAGE_DIR, x), os.listdir(POS_IMAGE_DIR))
    neg_file_path_list = map(lambda x: os.path.join(NEG_IMAGE_DIR, x), os.listdir(NEG_IMAGE_DIR))
    
    for index, pic in enumerate(pos_file_path_list):
        f_out = os.path.join(RESIZE_POS_IMAGE_DIR, '{}.{}'.format(index, IMG_TYPE))
        resize_image(pic, f_out, IMG_WIDTH, IMG_HEIGHT)
    for index, pic in enumerate(neg_file_path_list):
        f_out = os.path.join(RESIZE_NEG_IMAGE_DIR, '{}.{}'.format(index, IMG_TYPE))
        resize_image(pic, f_out, IMG_WIDTH, IMG_HEIGHT)
    
    pos_file_path_list = map(lambda x: os.path.join(RESIZE_POS_IMAGE_DIR, x), os.listdir(RESIZE_POS_IMAGE_DIR))
    neg_file_path_list = map(lambda x: os.path.join(RESIZE_NEG_IMAGE_DIR, x), os.listdir(RESIZE_NEG_IMAGE_DIR))
    
    train_file_list0, train_label_list0, test_file_list0, test_label_list0 = split_data(pos_file_path_list,
                                                                                        [1] * len(pos_file_path_list),
                                                                                        rate=0.5)
    train_file_list1, train_label_list1, test_file_list1, test_label_list1 = split_data(neg_file_path_list,
                                                                                        [-1] * len(neg_file_path_list),
                                                                                        rate=0.5)
    
    train_file_list = train_file_list0 + train_file_list1
    train_label_list = train_label_list0 + train_label_list1
    test_file_list = test_file_list0 + test_file_list1
    test_label_list = test_label_list0 + test_label_list1
    
    train_image_array = load_images(train_file_list, width=IMG_WIDTH, height=IMG_HEIGHT)
    train_label_array = np.array(train_label_list)
    test_image_array = load_images(test_file_list, width=IMG_WIDTH, height=IMG_HEIGHT)
    test_label_array = np.array(test_label_list)
    

    train_hist_array = get_lbp_data(train_image_array, hist_size=256, lbp_radius=1, lbp_point=8)
    test_hist_array = get_lbp_data(test_image_array, hist_size=256, lbp_radius=1, lbp_point=8)
    
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)  # SVC, NuSVC, SVR, NuSVR, OneClassSVM, LinearSVC, LinearSVR
    
    score = OneVsRestClassifier(svr_rbf, n_jobs=-1).fit(train_hist_array, train_label_array).score(test_hist_array,
                                                                                                   test_label_array)  # n_jobs是cpu数量, -1代表所有
    print score
    return score


if __name__ == '__main__':
    n = 10
    scores = []
    for i in range(n):
        s = main()
        scores.append(s)
    max_s = max(scores)
    min_s = min(scores)
    avg_s = sum(scores)/float(n)
    print '==========\nmax: %s\nmin: %s\navg: %s' % (max_s, min_s, avg_s)
