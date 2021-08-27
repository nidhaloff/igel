import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense ,Flatten ,Dropout , Conv1D , Conv1DTranspose,Conv2D,Conv2DTranspose,Conv3D,Conv3DTranspose,MaxPool1D,MaxPool2D,MaxPool3D,MaxPooling1D,MaxPooling2D,MaxPooling3D,AveragePooling1D,AveragePooling2D,AveragePooling3D,GlobalAveragePooling1D,GlobalAveragePooling2D,GlobalAveragePooling3D,GlobalMaxPooling1D,GlobalMaxPooling2D,GlobalMaxPooling3D,BatchNormalization,LayerNormalization
from keras.losses import SparseCategoricalCrossentropy, BCE, CategoricalCrossentropy , KLDivergence

import yaml


class CNN:
  def __init__(self,path):
    """
    The YAML file is parsed and all training parameters/ dataset arguments are set. 
    We stick with the defaults in case the user doesnt provide them

    """
    with open(path, 'rb') as f:
      conf = yaml.safe_load(f.read()) 
    self.batch_size= conf['model']['arguments'].get('batch_size',32)
    self.epochs = conf['model']['arguments'].get('epochs',20)
    self.validation_split = conf['model']['arguments'].get('validation_split',0.2)
    self.filename = conf['model']['arguments'].get('filename','train.csv')
    self.color_mode = conf['model']['arguments'].get('color_mode','rgb')
    self.target_size = conf['model']['arguments'].get('target_size',[256,256])
    self.target_size=tuple(self.target_size)
    self.class_mode = conf['model']['arguments'].get('class_mode','categorical')
    self.image_size = conf['model']['arguments'].get('image_size',(784,))
    

  def make_model(self,conf):
    """
    Iterating over all the layers mentioned by the user, we incrementally add them to the model
    All the layers supported by igel can be found here ---> https://keras.io/api/layers/
    We currently support 15+ layers 
    """
    model=keras.Sequential()
    for i in conf['model']['arguments']['model_layers'].items():
      x= i[1].get('type')
      #print(x)
      #if i[1].get('parameters'):
      size=i[1].get('parameters').get('size')
      Dense_activation=i[1].get('parameters').get('activation','sigmoid')
      Conv_activation=i[1].get('parameters').get('activation',None)
      filter_size=i[1].get('parameters').get('filter_size',3)
      stride_1d = i[1].get('parameters').get('stride',1)
      stride_2d= i[1].get('parameters').get('stride',[1,1])
      stride_3d= i[1].get('parameters').get('stride',[1,1,1])
      value= i[1].get('parameters').get('dropout_value',0.2)
      padding=i[1].get('parameters').get('padding','valid')
      kernel_size=i[1].get('parameters').get('kernel_size',1)
      kernel_size_2d=i[1].get('parameters').get('kernel_size',[1,1])
      kernel_size_3d=i[1].get('parameters').get('kernel_size',[1,1,1])
      pool_size= i[1].get('parameters').get('pool_size',2)
      pool_size_2d= i[1].get('parameters').get('pool_size',[2,2])
      pool_size_3d= i[1].get('parameters').get('pool_size',[2,2,2])
      stride_2d= tuple(stride_2d)
      stride_3d=tuple(stride_3d)
      kernel_size_2d= tuple(kernel_size_2d)
      kernel_size_3d= tuple(kernel_size_3d)
      pool_size_2d= tuple(pool_size_2d)
      pool_size_3d= tuple(pool_size_3d)
      
      if x == "Dense":
        model.add(keras.layers.Dense(units=size,activation=Dense_activation))
      elif x == "Dropout":
        model.add(keras.layers.Dropout(rate=value))
      elif x == "Flatten":
        model.add(keras.layers.Flatten(input_shape=self.image_size))  
      elif x == "Conv2D":
        model.add(keras.layers.Conv2D(filters=filter_size,strides=stride_2d,padding=padding,activation=Conv_activation,kernel_size=kernel_size_2d))
      elif x == "Conv1D":
        model.add(keras.layers.Conv1D(filters=filter_size,strides=stride_1d,padding=padding,activation=Conv_activation,kernel_size=kernel_size))
      elif x == "Conv3D":
        model.add(keras.layers.Conv3D(filters=filter_size,strides=stride_3d,padding=padding,activation=Conv_activation,kernel_size=kernel_size_3d))
      elif x == "MaxPooling1D":
        model.add(keras.layers.MaxPooling1D(pool_size=pool_size,padding=padding))
      elif x == "MaxPooling2D":
        model.add(keras.layers.MaxPooling2D(pool_size=pool_size_2d,padding=padding))
      elif x == "MaxPooling3D":
        model.add(keras.layers.MaxPooling3D(pool_size=pool_size_3d,padding=padding))
      elif x == "AveragePooling1D":
        model.add(keras.layers.AveragePooling1D(pool_size=pool_size,padding=padding))
      elif x == "AveragePooling2D":
        model.add(keras.layers.AveragePooling2D(pool_size=pool_size_2d,padding=padding))
      elif x == "AveragePooling3D":
        model.add(keras.layers.AveragePooling3D(pool_size=pool_size_3d,padding=padding))
      elif x == "GlobalMaxPooling1D":
        model.add(keras.layers.GlobalMaxPooling1D())
      elif x == "GlobalMaxPooling2D":
        model.add(keras.layers.GlobalMaxPooling2D())
      elif x == "GlobalMaxPooling3D":
        model.add(keras.layers.GlobalMaxPooling3D())
      elif x == "GlobalAveragePooling1D":
        model.add(keras.layers.GlobalAveragePooling1D())
      elif x == "GlobalAveragePooling2D":
        model.add(keras.layers.GlobalAveragePooling2D())
      elif x == "GlobalAveragePooling3D":
        model.add(keras.layers.GlobalAveragePooling3D())
      else:
        print("Please enter a valid layer") 
      print(model.summary())
    #model.summary()
    return model

  def generate_dataset(self):
    """
    Using The ImadeDataGenerator class allows us to create batches of data (even augment them), while just providing the paths of the 
    data in a csv file.
    """
    df=pd.read_csv(self.filename)
    datagen=ImageDataGenerator(rescale=1./255.,validation_split=self.validation_split)
    df['label']=df['label'].astype(str)
    self.train_generator=datagen.flow_from_dataframe(dataframe=df,class_mode=self.class_mode,color_mode=self.color_mode,directory=None,x_col=list(df.columns)[0],y_col=list(df.columns)[1],target_size=self.target_size,subset="training",batch_size=self.batch_size,seed=42,shuffle=True,validate_filenames=True)
    self.valid_generator=datagen.flow_from_dataframe(dataframe=df,class_mode=self.class_mode,color_mode=self.color_mode,directory=None,x_col=list(df.columns)[0],y_col=list(df.columns)[1],target_size=self.target_size,subset="validation",batch_size=self.batch_size,seed=42,shuffle=True,validate_filenames=True)

  def train(self,model,conf):
    """
    A common cause of error while running the training loop is wrong choice of loss function

    """
    callbacks = [keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5")]
    lossfn= conf['model']['arguments'].get('loss','SparseCategoricalCrossentropy')
    if lossfn == "SparseCategoricalCrossentropy":
      lossf=keras.losses.SparseCategoricalCrossentropy()
    elif lossfn == "BCE":
      lossf=keras.losses.BinaryCrossentropy()
    elif lossfn == "CategoricalCrossentropy":
      lossf= keras.losses.CategoricalCrossentropy()
    elif lossfn == "KLDivergence":
      lossf= keras.losses.KLDivergence()
    else:
      print("Please enter a valid loss function")
    model.compile(optimizer=keras.optimizers.SGD(),  # Optimizer
    loss=lossf,
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    print("Fit model on training data")
    history = model.fit(self.train_generator, epochs=self.epochs, callbacks=callbacks, validation_data=self.valid_generator)

#Testing
#cnn= CNN('/content/igel.yaml') 
#cnn.generate_dataset()
#model = cnn.make_model(conf)
#cnn.train(model,conf)


