# -*- coding: utf-8 -*-
# Li Liu

import os
import csv
import numpy as np
from skimage import feature as skif
from skimage import io, transform
import random

# 全局变量
DATASET_DIR = os.path.join(os.path.dirname(__file__), '../datasets')  # The directory for datasets
FER_DIR = os.path.join(DATASET_DIR, 'fer2013') #The directory for fer2013 dataset
CK_DIR = os.path.join(DATASET_DIR, 'Cohn-Kanade/cropped') #The directory for fer2013 dataset
CK_IMG = os.path.join(CK_DIR, 'images')
FER_IMG = os.path.join(FER_DIR, 'images')
IMG_TYPE = 'png'  # 图片类型

# Load images
def load_images(images_list, width, height,resize=True):
    data = np.zeros((len(images_list), width, height))  # Create array for image data storing
    for index,image in enumerate(images_list):
        if(resize == True):
            image = os.path.join(CK_DIR,'images',image)
        else:
            image = os.path.join(FER_DIR,'images',image)
        image_data = io.imread(image, as_grey=True)
        image_data = transform.resize(image_data, (width, height))
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
        arr=np.zeros(lbp_point,dtype=int)
        arr[0:i]=1
        rotate(arr,patterns)    
    patterns.sort()
    m={}
    m[0]=0
    num=1
    for p in patterns:
        m[p]=num
        num+=1
    m[pow(2,lbp_point)-1]=num
    return m

def replace(k,m):
    if m.get(k) == None:
        return max(list(m.values()))+1
    else:
        return m.get(k)

def get_lbp_with_region_feature(images_data, lbp_radius=1, lbp_point=8, row=6, column=6):
    uniform_map = assign_bin(lbp_point)
    fpf_replace=np.frompyfunc(replace,2,1)

    w = images_data.shape[1]
    h = images_data.shape[2]
    n_images = images_data.shape[0]

    bin_num = (lbp_point - 1) * lbp_point + 3
    hist = []

    for i in np.arange(n_images):

        # extract non-rotation-invariant uniform lbp feature from each image
        lbp = skif.local_binary_pattern(images_data[i], lbp_point, lbp_radius, 'nri_uniform')
        
        lbp = np.asarray(lbp)
        lbp = fpf_replace(lbp, uniform_map)
        lbp = lbp.reshape(row,int(h/row),column,int(w/column))
        lbp = np.swapaxes(lbp,1,2).reshape(row*column,-1)
        r = np.apply_along_axis(np.histogram,1,lbp,bins=np.arange(bin_num+1))
        hist.append(np.concatenate(r[:,0]))

    return hist


def main():
    '''
    ck_train_lists = map(lambda x: os.path.join(CK_DIR,'TrainList',x), os.listdir(os.path.join(CK_DIR,'TrainList')))
    ck_train_lists = list(ck_train_lists)
    ck_test_list = os.path.join(CK_DIR,'TestList/overall_test.csv')
    types=['U50','i4']
    for item in ck_train_lists[1:]:
        
        ind = item.split('_')
        mix_list = np.genfromtxt(item,dtype=types,delimiter=',',names='True')
        image_list = mix_list['True']
        data = load_images(image_list, 240, 240)
        hist = get_lbp_with_region_feature(data,2,8,6,6)
        feature_dir = os.path.join(CK_DIR,'LBPFeature/train/2_8_6_6')
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)
        print(os.path.join(feature_dir,'fold_'+ind[1]+'.csv'))
        featureFile = open(os.path.join(feature_dir,'fold_'+ind[1]+'.csv'),'w')
        writer = csv.writer(featureFile)
        writer.writerows(hist)

    mix_list = np.genfromtxt(ck_test_list,dtype=types,delimiter=',',names='True')
    image_list = mix_list['True']
    data = load_images(image_list, 240, 240)
    hist = get_lbp_with_region_feature(data,2,8,6,6)
    feature_dir = os.path.join(CK_DIR,'LBPFeature/test/2_8_6_6')
    if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)
    print(os.path.join(feature_dir,'test.csv'))
    featureFile = open(os.path.join(feature_dir,'test.csv'), 'w')
    writer = csv.writer(featureFile)
    writer.writerows(hist)
    '''
    fer_train_lists = map(lambda x: os.path.join(FER_DIR,'TrainList',x), os.listdir(os.path.join(FER_DIR,'TrainList')))
    fer_train_lists = list(fer_train_lists)
    fer_test_list = os.path.join(FER_DIR,'TestList/overall_test.csv')
    types=['U50','i4']
    for item in fer_train_lists[1:]:
        ind = item.split('_')
        mix_list = np.genfromtxt(item,dtype=types,delimiter=',',names='True')
        image_list = mix_list['True']
        data = load_images(image_list, 48,48, resize=False)
        hist = get_lbp_with_region_feature(data,1,8,2,2)
        feature_dir = os.path.join(FER_DIR,"LBPFeature/train/1_8_2_2")
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)
        print(os.path.join(feature_dir,"fold_"+ind[1]+".csv"))
        featureFile = open(os.path.join(feature_dir,"fold_"+ind[1]+".csv"), 'w')
        writer = csv.writer(featureFile)
        writer.writerows(hist)

    mix_list = np.genfromtxt(fer_test_list,dtype=types,delimiter=',',names='True')
    image_list = mix_list['True']
    data = load_images(image_list, 240, 240)
    hist = get_lbp_with_region_feature(data,1,8,2,2)
    feature_dir = os.path.join(FER_DIR,"LBPFeature/test/1_8_2_2")
    if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)
    print(os.path.join(feature_dir,"test.csv"))
    featureFile = open(os.path.join(feature_dir,"test.csv"), 'w')
    writer = csv.writer(featureFile)
    writer.writerows(hist)

if __name__ == '__main__':
    main()