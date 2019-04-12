# This script loops through the Cohn-Kanade data, filters out images without labels, and outputs a csv file containing image filenames and their labels

import os
from shutil import copy

img_dir = os.getcwd()+"\\Cohn-Kanade\\cohn-kanade-images"
emotion_dir = os.getcwd()+"\\Cohn-Kanade\\Emotion"
all_img_dir = os.getcwd()+"\\Cohn-Kanade\\all-images"
dst = """C:\\Users\\yalam\\Documents\\Spring 2019\\CSE 802\\Project\\Combined Datasets\\Cohn-Kanade\\all-images"""
all_samples = []

# Loop through each subject's folder
for file in os.listdir(os.fsencode(emotion_dir)):
    subfolder = os.fsdecode(file)

    # loop through each sequence in current subject's folder
    for new_folder in os.listdir(os.fsencode(emotion_dir+"\\"+subfolder)):

        printable_new_folder = os.fsdecode(new_folder)
        label_len = len(os.listdir(os.fsencode(emotion_dir+"\\"+subfolder+"\\"+printable_new_folder)))

        if label_len == 0:
            # A label doesn't exist for this sequence, so move on
            continue
            
        # Found a label file (Disregard the loop, it will only perform one iteration)
        for file in os.listdir(os.fsencode(emotion_dir+"\\"+subfolder+"\\"+printable_new_folder)):

            # Open the label file and append label to csv file
            with open(emotion_dir+"\\"+subfolder+"\\"+printable_new_folder+"\\"+os.fsdecode(file)) as label_file:
                label = 0.0
                for line in label_file:
                    label += float(line.strip())
                    
            # Copy these usable images to an output folder
            for img_file in os.listdir(os.fsencode(img_dir+"\\"+subfolder+"\\"+printable_new_folder)):
                src = img_dir+"\\"+subfolder+"\\"+printable_new_folder+"\\"+os.fsdecode(img_file)
                copy(src, dst)
                all_samples.append([os.fsdecode(img_file),float(line.strip())])

# Open a csv file and write the labels and image data
with open('labels.csv','w') as output:
    for line in all_samples:
        for elem in line:
            output.write(str(elem))
            output.write(",")
        output.write('\n')