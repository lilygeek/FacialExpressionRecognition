#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sklearn
import numpy as np
import time
import copy


# In[2]:


from sklearn.decomposition import PCA


# In[3]:


cohn_kanade_test = []
with open('datasets/Cohn-Kanade/cropped/LBPFeature/test/2_8_6_6/test.csv','r') as data_file:
    cohn_kanade_test = [[int(num) for num in line.strip().split(",")] for line in data_file.readlines()]


# In[4]:


cohn_kanade_train_1 = []
cohn_kanade_train_2 = []
cohn_kanade_train_3 = []
cohn_kanade_train_4 = []
cohn_kanade_train_5 = []
with open('datasets/Cohn-Kanade/cropped/LBPFeature/train/2_8_6_6/fold_1.csv','r') as data_file:
    cohn_kanade_train_1 = [[int(num) for num in line.strip().split(",")] for line in data_file.readlines()]
with open('datasets/Cohn-Kanade/cropped/LBPFeature/train/2_8_6_6/fold_2.csv','r') as data_file:
    cohn_kanade_train_2 = [[int(num) for num in line.strip().split(",")] for line in data_file.readlines()]
with open('datasets/Cohn-Kanade/cropped/LBPFeature/train/2_8_6_6/fold_3.csv','r') as data_file:
    cohn_kanade_train_3 = [[int(num) for num in line.strip().split(",")] for line in data_file.readlines()]
with open('datasets/Cohn-Kanade/cropped/LBPFeature/train/2_8_6_6/fold_4.csv','r') as data_file:
    cohn_kanade_train_4 = [[int(num) for num in line.strip().split(",")] for line in data_file.readlines()]
with open('datasets/Cohn-Kanade/cropped/LBPFeature/train/2_8_6_6/fold_5.csv','r') as data_file:
    cohn_kanade_train_5 = [[int(num) for num in line.strip().split(",")] for line in data_file.readlines()]


# In[5]:


cohn_kanade_test_labels = []
with open('datasets/Cohn-Kanade/cropped/TestList/overall_test.csv','r') as data_file:
    cohn_kanade_test_labels = [line.strip().split(",") for line in data_file.readlines()]


# In[6]:


cohn_kanade_train_1_labels = []
cohn_kanade_train_2_labels = []
cohn_kanade_train_3_labels = []
cohn_kanade_train_4_labels = []
cohn_kanade_train_5_labels = []
with open('datasets/Cohn-Kanade/cropped/TrainList/fold_1_train.csv','r') as data_file:
    cohn_kanade_train_1_labels = [line.strip().split(",") for line in data_file.readlines()]
with open('datasets/Cohn-Kanade/cropped/TrainList/fold_2_train.csv','r') as data_file:
    cohn_kanade_train_2_labels = [line.strip().split(",") for line in data_file.readlines()]
with open('datasets/Cohn-Kanade/cropped/TrainList/fold_3_train.csv','r') as data_file:
    cohn_kanade_train_3_labels = [line.strip().split(",") for line in data_file.readlines()]
with open('datasets/Cohn-Kanade/cropped/TrainList/fold_4_train.csv','r') as data_file:
    cohn_kanade_train_4_labels = [line.strip().split(",") for line in data_file.readlines()]
with open('datasets/Cohn-Kanade/cropped/TrainList/fold_5_train.csv','r') as data_file:
    cohn_kanade_train_5_labels = [line.strip().split(",") for line in data_file.readlines()]


# In[7]:


fer_test = []
with open('datasets/fer2013/LBPFeature/test/1_8_2_2/test.csv','r') as data_file:
    fer_test = [[int(num) for num in line.strip().split(",")] for line in data_file.readlines()]


# In[8]:


fer_train_1 = []
fer_train_2 = []
fer_train_3 = []
fer_train_4 = []
fer_train_5 = []
with open('datasets/fer2013/LBPFeature/train/1_8_2_2/fold_1.csv','r') as data_file:
    fer_train_1 = [[int(num) for num in line.strip().split(",")] for line in data_file.readlines()]
with open('datasets/fer2013/LBPFeature/train/1_8_2_2/fold_2.csv','r') as data_file:
    fer_train_2 = [[int(num) for num in line.strip().split(",")] for line in data_file.readlines()]
with open('datasets/fer2013/LBPFeature/train/1_8_2_2/fold_3.csv','r') as data_file:
    fer_train_3 = [[int(num) for num in line.strip().split(",")] for line in data_file.readlines()]
with open('datasets/fer2013/LBPFeature/train/1_8_2_2/fold_4.csv','r') as data_file:
    fer_train_4 = [[int(num) for num in line.strip().split(",")] for line in data_file.readlines()]
with open('datasets/fer2013/LBPFeature/train/1_8_2_2/fold_5.csv','r') as data_file:
    fer_train_5 = [[int(num) for num in line.strip().split(",")] for line in data_file.readlines()]


# In[9]:


fer_test_labels = []
with open('datasets/fer2013/TestList/overall_test.csv','r') as data_file:
    fer_test_labels = [line.strip().split(",") for line in data_file.readlines()]


# In[10]:


fer_train_1_labels = []
fer_train_2_labels = []
fer_train_3_labels = []
fer_train_4_labels = []
fer_train_5_labels = []
with open('datasets/fer2013/TrainList/fold_1_train.csv','r') as data_file:
    fer_train_1_labels = [line.strip().split(",") for line in data_file.readlines()]
with open('datasets/fer2013/TrainList/fold_1_train.csv','r') as data_file:
    fer_train_2_labels = [line.strip().split(",") for line in data_file.readlines()]
with open('datasets/fer2013/TrainList/fold_1_train.csv','r') as data_file:
    fer_train_3_labels = [line.strip().split(",") for line in data_file.readlines()]
with open('datasets/fer2013/TrainList/fold_1_train.csv','r') as data_file:
    fer_train_4_labels = [line.strip().split(",") for line in data_file.readlines()]
with open('datasets/fer2013/TrainList/fold_1_train.csv','r') as data_file:
    fer_train_5_labels = [line.strip().split(",") for line in data_file.readlines()]


# In[11]:


len(fer_train_1[0])


# In[13]:


pca = PCA(n_components=65, svd_solver='randomized').fit(fer_train_1)


# In[14]:


reduced_data_pca = pca.fit_transform(fer_train_1)


# In[15]:


reduced_data_pca.shape


# In[16]:


len(fer_train_1)


# In[17]:


# import the necessary module
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# In[18]:


#create an object of the type GaussianNB
gnb = GaussianNB()

#train the algorithm on training data and predict using the testing data
pred = gnb.fit(fer_train_1, [row[1] for row in fer_train_1_labels]).predict(fer_test)
#print(pred.tolist())

#print the accuracy score of the model
print("Naive-Bayes accuracy : ", accuracy_score([row[1] for row in fer_test_labels], pred, normalize = True))


# In[45]:


all_train = copy.deepcopy(fer_train_1)
all_train.extend(fer_train_2)
all_train.extend(fer_train_3)
all_train.extend(fer_train_4)
all_train.extend(fer_train_5)

all_train_labels = copy.deepcopy(fer_train_1_labels)
all_train_labels.extend(fer_train_2_labels)
all_train_labels.extend(fer_train_3_labels)
all_train_labels.extend(fer_train_4_labels)
all_train_labels.extend(fer_train_5_labels)
all_train_labels = [row[1] for row in all_train_labels]

all_test_labels = [row[1] for row in fer_test_labels]

#create an object of the type GaussianNB
start = time.time()
gnb = GaussianNB()

#train the algorithm on training data and predict using the testing data
pred = gnb.fit(all_train, all_train_labels).predict(fer_test)
end = time.time()

#print the accuracy score of the model
print("Naive-Bayes accuracy (full data): ", accuracy_score(all_test_labels, pred, normalize = True))
print("Time Taken: ", end - start, " seconds")


# In[ ]:





# In[46]:


all_train = copy.deepcopy(cohn_kanade_train_1)
all_train.extend(cohn_kanade_train_2)
all_train.extend(cohn_kanade_train_3)
all_train.extend(cohn_kanade_train_4)
all_train.extend(cohn_kanade_train_5)

all_train_labels = copy.deepcopy(cohn_kanade_train_1_labels)
all_train_labels.extend(cohn_kanade_train_2_labels)
all_train_labels.extend(cohn_kanade_train_3_labels)
all_train_labels.extend(cohn_kanade_train_4_labels)
all_train_labels.extend(cohn_kanade_train_5_labels)
all_train_labels = [row[1] for row in all_train_labels]

all_test_labels = [row[1] for row in cohn_kanade_test_labels]

#create an object of the type GaussianNB
gnb = GaussianNB()

#train the algorithm on training data and predict using the testing data
pred = gnb.fit(all_train, all_train_labels).predict(cohn_kanade_test)

#print the accuracy score of the model
print("Naive-Bayes accuracy (full data): ", accuracy_score(all_test_labels, pred, normalize = True))


# In[ ]:





# In[75]:


all_train_labels = copy.deepcopy(fer_train_1_labels)
all_train_labels.extend(fer_train_2_labels)
all_train_labels.extend(fer_train_3_labels)
all_train_labels.extend(fer_train_4_labels)
all_train_labels.extend(fer_train_5_labels)
all_train_labels = [row[1] for row in all_train_labels]
all_test_labels = [row[1] for row in fer_test_labels]

for components in range(1, 75, 10):
    all_test = copy.deepcopy(fer_test)

    all_train = copy.deepcopy(fer_train_1)
    all_train.extend(fer_train_2)
    all_train.extend(fer_train_3)
    all_train.extend(fer_train_4)
    all_train.extend(fer_train_5)

    pca = PCA(n_components=components).fit(all_train)
    all_train = pca.fit_transform(all_train)
    all_test = pca.fit_transform(all_test)

    #create an object of the type GaussianNB
    start = time.time()
    gnb = GaussianNB()

    #train the algorithm on training data and predict using the testing data
    pred = gnb.fit(all_train, all_train_labels).predict(all_test)
    end = time.time()

    #print the accuracy score of the model
    print("FER Naive-Bayes accuracy (", components, " components): ", accuracy_score(all_test_labels, pred, normalize = True))
    print("Time Taken: ", end - start, " seconds")


# In[40]:


all_train_labels = copy.deepcopy(cohn_kanade_train_1_labels)
all_train_labels.extend(cohn_kanade_train_2_labels)
all_train_labels.extend(cohn_kanade_train_3_labels)
all_train_labels.extend(cohn_kanade_train_4_labels)
all_train_labels.extend(cohn_kanade_train_5_labels)
all_train_labels = [row[1] for row in all_train_labels]
all_test_labels = [row[1] for row in cohn_kanade_test_labels]

for components in range(1, 55, 10):
    all_train = copy.deepcopy(cohn_kanade_train_1)
    all_train.extend(cohn_kanade_train_2)
    all_train.extend(cohn_kanade_train_3)
    all_train.extend(cohn_kanade_train_4)
    all_train.extend(cohn_kanade_train_5)

    all_test = copy.deepcopy(cohn_kanade_test)

    pca = PCA(n_components=components).fit(all_train)
    all_train = pca.fit_transform(all_train)
    all_test = pca.fit_transform(all_test)

    #create an object of the type GaussianNB
    start = time.time()
    gnb = GaussianNB()

    #train the algorithm on training data and predict using the testing data
    pred = gnb.fit(all_train, all_train_labels).predict(all_test)
    end = time.time()

    #print the accuracy score of the model
    print("COHN KANADE Naive-Bayes accuracy (", components, " components): ", accuracy_score(all_test_labels, pred, normalize = True))
    print("Time Taken: ", end - start, " seconds")


# In[44]:


all_train_labels = copy.deepcopy(cohn_kanade_train_1_labels)
all_train_labels.extend(cohn_kanade_train_2_labels)
all_train_labels.extend(cohn_kanade_train_3_labels)
all_train_labels.extend(cohn_kanade_train_4_labels)
all_train_labels.extend(cohn_kanade_train_5_labels)
all_train_labels = [row[1] for row in all_train_labels]
all_test_labels = [row[1] for row in cohn_kanade_test_labels]

components = 196
all_train = copy.deepcopy(cohn_kanade_train_1)
all_train.extend(cohn_kanade_train_2)
all_train.extend(cohn_kanade_train_3)
all_train.extend(cohn_kanade_train_4)
all_train.extend(cohn_kanade_train_5)

all_test = copy.deepcopy(cohn_kanade_test)

pca = PCA(n_components=components).fit(all_train)
all_train = pca.fit_transform(all_train)
all_test = pca.fit_transform(all_test)

#create an object of the type GaussianNB
start = time.time()
gnb = GaussianNB()

#train the algorithm on training data and predict using the testing data
pred = gnb.fit(all_train, all_train_labels).predict(all_test)
end = time.time()

#print the accuracy score of the model
print("COHN KANADE Naive-Bayes accuracy (", components, " components): ", accuracy_score(all_test_labels, pred, normalize = True))
print("Time Taken: ", end - start, " seconds")


# In[ ]:




