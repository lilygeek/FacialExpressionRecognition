#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
from shutil import copy


# In[2]:


all_labels = []
with open("labels.csv") as labels:
    for line in labels.readlines():
        cur_line = line.strip().split(",")[:-1]
        all_labels.append(cur_line)


# In[3]:


all_labels


# In[4]:


def same_set(row1, row2):
    return row1[0][0:4] == row2[0][0:4] and row1[0][5:8] == row2[0][5:8] and row1[1] == row2[1]


# In[5]:


split = []
for row in all_labels:
    if len(split) == 0:
        split.append([row])
    else:
        not_added = True
        for section in split:
            if same_set(section[0], row):
                section.append(row)
                not_added = False
        if not_added:
            split.append([row])


# In[6]:


len(split)


# In[7]:


len(split[0])


# In[8]:


split[0][0]


# In[9]:


filtered_data = []
for same_set in split:
    if len(same_set) > 5:
        # TODO
        filtered_data.append([same_set[0][0], "NEUTRAL"])
        filtered_data.append([same_set[1][0], "NEUTRAL"])
        filtered_data.append([same_set[2][0], "NEUTRAL"])
        
        filtered_data.append(same_set[-1])
        filtered_data.append(same_set[-2])
        if same_set[-3] not in filtered_data:
            filtered_data.append(same_set[-3])
        #if same_set
    else:
        filtered_data.extend(same_set)


# In[10]:


filtered_data


# In[11]:


dst = os.getcwd()+"\\filtered-all-images"
src = os.getcwd()+"\\all-images"
just_img_names = [i[0] for i in filtered_data]


# In[22]:


for file in os.listdir(os.fsencode(src)):
    pic = os.fsdecode(file)
    if pic in just_img_names:
        this_src = src+"\\"+pic
        copy(this_src, dst)


# In[23]:


len(filtered_data)


# In[25]:


with open("filtered_labels.csv", "w") as output:
    for line in filtered_data:
        for elem in line:
            output.write(str(elem))
            output.write(",")
        output.write('\n')


# In[26]:


import random
random.shuffle(filtered_data)


# In[42]:


overall_test = filtered_data[:196]
with open("labels-split\overall_test.csv", "w") as output:
    for line in overall_test:
        for elem in line:
            output.write(str(elem))
            output.write(",")
        output.write('\n')
overall_train = filtered_data[196:-1]


# In[32]:


len(overall_train)


# In[33]:


fold_1 = overall_train[:353]
fold_2 = overall_train[353:706]
fold_3 = overall_train[706:1059]
fold_4 = overall_train[1059:1412]
fold_5 = overall_train[1412:]


# In[45]:


fold_1_train = fold_1[:35]
fold_1_test = fold_1[35:]
fold_2_train = fold_2[:35]
fold_2_test = fold_2[35:]
fold_3_train = fold_3[:35]
fold_3_test = fold_3[35:]
fold_4_train = fold_4[:35]
fold_4_test = fold_4[35:]
fold_5_train = fold_5[:35]
fold_5_test = fold_5[35:]


# In[60]:


for i,fold in enumerate([fold_1_train, fold_2_train, fold_3_train, fold_4_train, fold_5_train]):
    with open("labels-split\\fold_{}_test.csv".format(i+1), "w") as output:
        for line in fold:
            for elem in line:
                output.write(str(elem))
                output.write(",")
            output.write('\n')


# In[61]:


for i,fold in enumerate([fold_1_test, fold_2_test, fold_3_test, fold_4_test, fold_5_test]):
    with open("labels-split\\fold_{}_train.csv".format(i+1), "w") as output:
        for line in fold:
            for elem in line:
                output.write(str(elem))
                output.write(",")
            output.write('\n')


# In[ ]:




