import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense ,Flatten ,Dropout , Conv1D , Conv1DTranspose,Conv2D,Conv2DTranspose,Conv3D,Conv3DTranspose,MaxPool1D,MaxPool2D,MaxPool3D,MaxPooling1D,MaxPooling2D,MaxPooling3D,AveragePooling1D,AveragePooling2D,AveragePooling3D,GlobalAveragePooling1D,GlobalAveragePooling2D,GlobalAveragePooling3D,GlobalMaxPooling1D,GlobalMaxPooling2D,GlobalMaxPooling3D,BatchNormalization,LayerNormalization

class CNN:
    model_list=[]
    def __init__(self,batch_size=32,epochs=20,validation_split=0.2,filename="train.csv",color_mode='rgb',class_mode='categorical',target_size=(256,256)):
        self.batch_size=batch_size
        self.epochs=epochs
        self.validation_split=validation_split
        self.filename=filename
        self.color_mode=color_mode #options are grayscale,rgb (default)
        self.class_mode=class_mode #categorical, sparse, multi-input
        self.target_size=target_size
        
 
    
    def make_model(self,*args):
        for i in args:
            self.model_list.append(i)
        model=keras.Sequential(self.model_list)
        return model

    def generate_dataset(self):
        df=pd.read_csv(self.filename)
        datagen=ImageDataGenerator(rescale=1./255.,validation_split=self.validation_split)
        df['label']=df['label'].astype(str)
        self.train_generator=datagen.flow_from_dataframe(dataframe=df,class_mode=self.class_mode,color_mode=self.color_mode,directory=None,x_col=list(df.columns)[0],y_col=list(df.columns)[1],target_size=self.target_size,subset="training",batch_size=self.batch_size,seed=42,shuffle=True,validate_filenames=True)
        self.valid_generator=datagen.flow_from_dataframe(dataframe=df,class_mode=self.class_mode,color_mode=self.color_mode,directory=None,x_col=list(df.columns)[0],y_col=list(df.columns)[1],target_size=self.target_size,subset="validation",batch_size=self.batch_size,seed=42,shuffle=True,validate_filenames=True)
        

    
    def train(self,model):
        callbacks = [keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5")]
        model.compile(optimizer=keras.optimizers.SGD(),  # Optimizer
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )
        print("Fit model on training data")
        history = model.fit(self.train_generator, epochs=self.epochs, callbacks=callbacks, validation_data=self.valid_generator)
 
cnn= CNN(color_mode="grayscale",class_mode='sparse',target_size=(28,28)) #Parsed YAML dataset argument is fed here
cnn.generate_dataset()
model=cnn.make_model(Flatten(),Dense(32,activation='sigmoid'),Dense(10,activation='softmax')) # Parsed YAML model architecture argument is fed here
cnn.train(model)
