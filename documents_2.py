from numpy import array
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import cv2
import numpy as np


# extract features from each photo in the directory
def extract_vgg_features(images):
    # load the model
    model = VGG16()
    # re-structure the model
    model.layers.pop()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # model = VGG16(weights='imagenet', include_top=False)
    # summarize

    features = []
    for image in images:
        # print(image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.resize(image, (224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # print(image.shape)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # get features
        feature = model.predict(image, verbose=0)
        feature = feature.reshape(4096)
        # get image ids
        # print(feature.shape)
        features.append(array(feature))
    # print(features)
    # print(array(features).shape)
    return array(features)


def recover_document(images):
    features = extract_vgg_features(images)
    model = build_model()
    features = features.reshape((1, features.shape[0], features.shape[1]))
    # print(model.input_shape)
    prediction = model.predict(features)
    labels = np.argmax(prediction, axis=2)
    return labels[0]


def build_model():
    # feature extractor model
    inputs1 = Input(shape=(4, 4096))
    # fe1 = Dense(256, activation='relu')(inputs1)
    # fe2 = Dropout(0.1)(fe1)

    # language model 32
    decoder1 = LSTM(32, dropout=0.1, recurrent_dropout=0.1, return_sequences=True)(inputs1)
    # decoder1 = Bidirectional(LSTM(32, dropout=0.5, recurrent_dropout=0.2, return_sequences=True))(inputs1)
    # , dropout=0.6, recurrent_dropout=0.1,

    # #fully connected
    # decoder2 = TimeDistributed(Dense(32, activation='relu'))(decoder1)
    # bn = BatchNormalization()(decoder2)
    # do = Dropout(0.4)(bn)

    outputs = TimeDistributed(Dense(4, activation='softmax'))(decoder1)
    # outputs = Dense(4, activation='softmax')(decoder1)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1], outputs=outputs)
    model.load_weights('documents_2.hdf5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize model
    return model


def train_model(model):
    # model.fit_generator(data, ytrain, epochs=20)
    batch_size = 32
    model.fit_generator(data_generator(train_features, batch_size),
                        # steps_per_epoch=975//batch_size,
                        steps_per_epoch=100,
                        epochs=20,
                        validation_data=data_generator(test_features, batch_size),
                        # validation_steps=325//batch_size)
                        validation_steps=20,
                        callbacks=callbacks_list)


# stuff to run always here such as class/def
def main():
    pass


if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main()
