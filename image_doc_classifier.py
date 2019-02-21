from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Flatten
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import cv2
import numpy as np

def is_image(images):
  model = build_model()
  probs = 0
  for image in images:
    im = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    im = cv2.resize(im, (150,150))
    im = im.astype(np.float64, casting='unsafe')
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen.standardize(im)
    im = im.reshape((1, im.shape[0], im.shape[1], im.shape[2]))
    probs += model.predict(im)[0][0]
  probs /= len(images)
  if probs > 0.5:
    return True
  return False
    
    

def build_model():
  model = Sequential()
  model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  
  model.add(Conv2D(32, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  
  model.add(Conv2D(64, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  
  model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
  model.add(Dense(64))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(1))
  model.add(Activation('sigmoid'))
  
  model.load_weights('doc_image_weights.hdf5')
  model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
  return model
  
def train_model(model):
  batch_size = 64
  # this is the augmentation configuration we will use for training
  train_datagen = ImageDataGenerator(
          rescale=1./255,
          shear_range=0.2,
          zoom_range=0.2,
          horizontal_flip=True)
  
  # this is a generator that will read pictures found in
  # subfolers of 'data/train', and indefinitely generate
  # batches of augmented image data
  train_generator = train_datagen.flow_from_directory(
          'dataset/train',  # this is the target directory
          target_size=(150, 150),  # all images will be resized to 150x150
          batch_size=batch_size,
          class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels
  
  
  # this is the augmentation configuration we will use for testing:
  # only rescaling
  test_datagen = ImageDataGenerator(rescale=1./255)
  
  # this is a similar generator, for validation data
  validation_generator = test_datagen.flow_from_directory(
          'dataset/validation',
          target_size=(150, 150),
          batch_size=batch_size,
          class_mode='binary')
          
  # fit model and save weights
  filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
  callbacks_list = [checkpoint]
          
  model.fit_generator(
          train_generator,
          epochs=10,
          validation_data=validation_generator,
          steps_per_epoch=80505//batch_size,
          validation_steps=20160//batch_size,
          callbacks=callbacks_list)
        
# stuff to run always here such as class/def
def main():
    pass

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()

        