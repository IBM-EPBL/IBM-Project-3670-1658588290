#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Part 1 - Building the CNN
#importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import optimizers


# In[2]:


# Initialing the CNN
classifier = Sequential()


# In[3]:


# Step 1 - Convolution Layer
classifier.add(Conv2D(32, (3,  3), input_shape = (64, 64, 3), activation = 'relu'))


# In[4]:


#step 2 - Pooling
classifier.add(MaxPooling2D(pool_size =(2,2)))


# In[5]:


# Adding second convolution layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))


# In[6]:


#Step 3 - Flattening
classifier.add(Flatten())


# In[13]:


#Step 4 - Full Connection
classifier.add(Dense(256, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(2, activation = 'softmax'))


# In[14]:


#Compiling The CNN
classifier.compile(
              optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])


# In[15]:


#Part 2 Fittting the CNN to the image
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


# In[16]:


test_datagen = ImageDataGenerator(rescale=1./255)


# In[17]:


training_set = train_datagen.flow_from_directory(
        '.\\Dataset\\train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')


# In[18]:


test_datagen = ImageDataGenerator(rescale=1./255)


# In[19]:


training_set.class_indices


# In[20]:


test_set = test_datagen.flow_from_directory(
        '.\\Dataset\\test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')


# In[22]:


model = classifier.fit_generator(training_set,
                    steps_per_epoch=len(training_set),
                    validation_data=test_set,
                    validation_steps=len(training_set),
                    epochs=10)


# In[25]:


#Saving the model
import h5py
classifier.save('Trained_CNN_Model.h5')


# In[26]:


print(model.history.keys())


# In[27]:


import matplotlib.pyplot as plt


# In[29]:


# summarize history for accuracy
plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[30]:


# summarize history for loss
plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

