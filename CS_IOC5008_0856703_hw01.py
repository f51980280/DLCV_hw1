#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import glob
import os
import numpy as np
import tensorflow as tf 
import keras
import keras.backend.tensorflow_backend as KTF
import tensorflow.keras.layers as Layers
import tensorflow.keras.optimizers as Optimizer
from keras.utils import np_utils
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import MaxPooling2D 
from keras.layers import Dense,Dropout,Input,BatchNormalization,Activation,Conv2D
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from sklearn.utils import shuffle
from PIL import Image

classes_labels = {'bedroom':0,'coast':1,'forest':2,'highway':3,'insidecity':4,'kitchen':5,'livingroom':6,'mountain':7,'office':8,'opencountry':9,'street':10,'suburb':11,'tallbuilding':12}
category={0:'bedroom',1:'coast',2:'forest',3:'highway',4:'insidectiy',5:'kitchen',6:'livingroom',7:'mountain',8:'office',9:'opencountry',10:'street',11:'suburb',12:'tallbuilding'}


# In[ ]:


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[ ]:


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


# In[ ]:


X_size=150
Y_size=150
num=0
X_train=np.zeros((X_size,Y_size))
Y_train=np.zeros(1)

for folders in glob.glob(r'C:\Users\Wei\Desktop\DL&CV\cs-ioc5008-hw1\dataset\dataset\train\*'):
    print(folders)
    label=os.path.basename(folders)
    print(label)
    for filename in os.listdir(folders):        
        img_dir=os.path.join(folders, filename)
        Img=Image.open(img_dir)
        test=Img.resize((X_size,Y_size),Image.BILINEAR)
        test=np.array(test,dtype=float)/255
        X_train=np.append(X_train,test)        
        Y_train = np.vstack((Y_train,classes_labels[label]))       
        print(num)
        num+=1

Y_train=np.delete(Y_train,0,axis=0)
X_train=X_train[X_size*Y_size:]
X_train=X_train.reshape(num,X_size,Y_size,1)
X,y=shuffle(X_train,Y_train,random_state=817328462)
X_4D=X.reshape(X.shape[0],X_size,Y_size,1).astype('float32')
y_OneHot = np_utils.to_categorical(y)


# In[ ]:


model=tf.keras.Sequential()
model.add(Layers.Conv2D(220,kernel_size=(3,3),activation='relu',input_shape=(150,150,1)))
model.add(Layers.Conv2D(180,kernel_size=(3,3),kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),activation='relu'))
model.add(Layers.MaxPool2D(5,5))
model.add(Layers.Dropout(rate=0.1))
model.add(Layers.Conv2D(180,kernel_size=(3,3),kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),activation='relu'))
model.add(Layers.Dropout(rate=0.2))
model.add(Layers.Conv2D(140,kernel_size=(3,3),activation='relu'))
model.add(Layers.Dropout(rate=0.2))
model.add(Layers.Conv2D(100,kernel_size=(3,3),activation='relu'))
model.add(Layers.Dropout(rate=0.2))
model.add(Layers.Conv2D(50,kernel_size=(3,3),activation='relu'))
model.add(Layers.Dropout(rate=0.2))
model.add(Layers.MaxPool2D(5,5))
model.add(Layers.Dropout(rate=0.5))
model.add(Layers.Flatten())
model.add(Layers.Dense(180,activation='relu'))
model.add(Layers.Dense(100,activation='relu'))
model.add(Layers.Dense(50,activation='relu'))
model.add(Layers.Dropout(rate=0.5))
model.add(Layers.Dense(13,activation='softmax'))

model.compile(optimizer=Optimizer.Adam(lr=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()


# In[ ]:


model.fit(X_4D,y,batch_size=32,epochs=1000,validation_split=0.05,verbose=1,shuffle=True)


# In[ ]:


X_size=150
Y_size=150
num=0

X_test = np.zeros((X_size,Y_size))

for folders in glob.glob(r'C:\Users\Wei\Desktop\DL&CV\cs-ioc5008-hw1\dataset\dataset\test\*'):      
    Img=Image.open(folders)
    test=Img.resize((X_size,Y_size),Image.BILINEAR)
    test=np.array(test, dtype=float)/255
    X_test=np.append(X_test,test)       
    print(num)
    num+=1

X_test=X_test[X_size*Y_size:]
X_test=X_test.reshape(num,X_size,Y_size,1)


# In[ ]:


list=[]
for filename in glob.glob(r'C:\Users\Wei\Desktop\DL&CV\cs-ioc5008-hw1\dataset\dataset\test\*'):
    temp=os.path.basename(filename)
    list.append(os.path.splitext(temp)[0])


# In[ ]:


ANS = model.predict_classes(X_test)
ans=[]
for i in range (0,1041):
    print(ANS[i])
    ans.append(category[ANS[i]])


# In[24]:


f=open("submission.csv", "w")  ##寫檔
f.write("{},{}\n".format("Id", "label"))
for x in zip(list, ans):
    f.write("{},{}\n".format(x[0], x[1]))
f.close()


# In[ ]:





# In[ ]:




