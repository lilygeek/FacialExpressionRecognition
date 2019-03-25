# -*- coding: utf-8 -*-
import csv
import os
from PIL import Image
import numpy as np


csv_path = './datasets/fer2013/fer2013.csv'
img_path = './datasets/fer2013/img'
num = 1
with open(csv_path) as f:
    csvr = csv.reader(f)
    for i, record in enumerate(csvr):
        if i == 0:
            continue
        label = record[0]
        pixel = record[1]
        pixel = np.asarray([float(p) for p in pixel.split()]).reshape(48, 48)
        subfolder = os.path.join(img_path, label)
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        im = Image.fromarray(pixel).convert('L')
        image_name = os.path.join(subfolder, '{:05d}.jpg'.format(i))
        print(image_name)
        im.save(image_name)