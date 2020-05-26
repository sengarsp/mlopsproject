#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist
dataset = mnist.load_data('mymnist.db')


# In[2]:


train , test = dataset
X_train , y_train = train
X_test , y_test = test
img1 = X_train[7]


# In[3]:


import cv2
X_train_1d = X_train.reshape(-1 , 28*28)
X_test_1d = X_test.reshape(-1 , 28*28)


# In[4]:


X_train = X_train_1d.astype('float32')
X_test = X_test_1d.astype('float32')


# In[5]:


from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
y_train_cat = to_categorical(y_train)
model = Sequential()


# In[6]:


model.add(Dense(units=512, input_dim=28*28, activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.summary()


# In[7]:


model.compile(optimizer=Adam(), loss='categorical_crossentropy', 
             metrics=['accuracy']
             )
h = model.fit(X_train, y_train_cat, epochs=10)


# In[8]:


model.save("output.txt")


# In[ ]:




