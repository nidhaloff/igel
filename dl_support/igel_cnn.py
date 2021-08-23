import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
try:
  df=pd.read_csv(r".\train.csv")
except:
  print("Dataset not found. Make sure it's labelled as train.csv")

class Cnn : 
  def __init__(self,image_size =(180,180),batch_size = 32,epochs=50,validation_split=0.2,num_classes=2):
      self.image_size=image_size
      self.batch_size=batch_size
      self.epochs=epochs
      self.validation_split=validation_split
      self.num_classes=len(pd.unique(df['label']))

  def generate_dataset(self):
      df=pd.read_csv(r".\train.csv")
      datagen=ImageDataGenerator(rescale=1./255)
      train_generator=datagen.flow_from_dataframe(dataframe=df,directory="./train/",x_col="id",y_col="label",subset="training",batch_size=32,seed=42,shuffle=True,class_mode="categorical",target_size=(32,32))
      valid_generator=datagen.flow_from_dataframe(dataframe=traindf,directory="./train/",x_col="id",y_col="label",subset="validation",batch_size=32,seed=42,shuffle=True,class_mode="categorical",target_size=(32,32))
      return train_generator,valid_generator
  
  
  def make_model(self,input_shape, num_classes):
      inputs = keras.Input(shape=input_shape)
      data_augmentation = keras.Sequential([layers.RandomFlip("horizontal"),layers.RandomRotation(0.1),])
      # Image augmentation block
      x = data_augmentation(inputs)
      

      # Entry block
      x = layers.Rescaling(1.0 / 255)(x)
      x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
      x = layers.BatchNormalization()(x)
      x = layers.Activation("relu")(x)

      x = layers.Conv2D(64, 3, padding="same")(x)
      x = layers.BatchNormalization()(x)
      x = layers.Activation("relu")(x)

      previous_block_activation = x  # Set aside residual

      for size in [128, 256, 512, 728]:
          x = layers.Activation("relu")(x)
          x = layers.SeparableConv2D(size, 3, padding="same")(x)
          x = layers.BatchNormalization()(x)

          x = layers.Activation("relu")(x)
          x = layers.SeparableConv2D(size, 3, padding="same")(x)
          x = layers.BatchNormalization()(x)

          x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

          # Project residual
          residual = layers.Conv2D(size, 1, strides=2, padding="same")(
              previous_block_activation
          )
          x = layers.add([x, residual])  # Add back residual
          previous_block_activation = x  # Set aside next residual

      x = layers.SeparableConv2D(1024, 3, padding="same")(x)
      x = layers.BatchNormalization()(x)
      x = layers.Activation("relu")(x)

      x = layers.GlobalAveragePooling2D()(x)
      if num_classes == 2:
          activation = "sigmoid"
          units = 1
      else:
          activation = "softmax"
          units = num_classes

      x = layers.Dropout(0.5)(x)
      outputs = layers.Dense(units, activation=activation)(x)
      return keras.Model(inputs, outputs)

  def train(self):
      callbacks = [keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5")]
      model.compile(
      optimizer=keras.optimizers.Adam(1e-3),
      loss="binary_crossentropy",
      metrics=["accuracy"])
      model.fit_generator(train_generator, epochs=epochs, callbacks=callbacks, validation_data=valid_generator)



cnn=Cnn()

dataset=cnn.generate_dataset()   
model = cnn.make_model(input_shape=cnn.image_size + (3,), num_classes=cnn.num_classes)
keras.utils.plot_model(model, show_shapes=True)
cnn.train() 