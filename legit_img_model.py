# from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
from keras import regularizers


class Network:
    def __init__(self):
        self.weight_decay = 0.0005
        self.x_shape = [100, 100, 1]  # i resized all examples to this size

        self.model = self.build_model()
        # self.model.load_weights('weights.h5')
        self.model.load_weights('weights2_img.h5')

    def build_model(self):
        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(3, 3)))

        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(32, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.4))
        model.add(Dense(2))
        model.add(Activation('softmax'))
        return model

    def predict(self, X):
        return self.model.predict(X)

    def fit2(self, X, y, X_val=None, y_val=None):
        # training parameters
        batch_size = 256
        maxepoches = 50
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 10

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))

        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        # optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        self.model.fit(X, y, validation_data=(X_val, y_val), epochs=maxepoches,
                       batch_size=batch_size, shuffle=True, callbacks=[reduce_lr], verbose=1)

    def fit(self, X, y, X_val=None, y_val=None):
        # training parameters
        batch_size = 64
        maxepoches = 50
        learning_rate = 0.001
        lr_decay = 1e-6
        lr_drop = 10

        def lr_scheduler(epoch):
            self.model.save_weights('weights_3(new).h5')
            return learning_rate * (0.5 ** (epoch // lr_drop))

        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        # optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X)

        historytemp = self.model.fit_generator(datagen.flow(X, y, batch_size=batch_size),
                                               steps_per_epoch=X.shape[0] // batch_size,
                                               epochs=maxepoches,
                                               validation_data=(X_val, y_val), callbacks=[reduce_lr], verbose=1)

# cwd = 'C:/Users/Rani/Desktop/Deep Project/project/'
# # dataset_path = cwd + 'image_dataset/'
# #
# # X_train = np.load(dataset_path+'X_train.npy')
# # X_val = np.load(dataset_path+'X_val.npy')
# # y_train = np.load(dataset_path+'y_train.npy')
# # y_val = np.load(dataset_path+'y_val.npy')
# #
# #
# # # cv2.imshow('image X[0]', X_val[32])
# # # print(y_val[32])
# # # cv2.waitKey(0)
# # # cv2.destroyAllWindows()
# # # exit()
# #
# # network = Network()
# #
# # true_indices = (y_val[:,1]==1)
# # true_x = X_val[true_indices]
# # preds = network.model.predict(true_x)
# #
# # for i in [  5,   27,   33,   61,   81,   94,  105,  111,  115,  117,  120,
# #         144,  166,  203,  210,  212,  219,  224,  243,  273]:
# #     cv2.imshow('image X[0]', X_val[true_indices][i])
# #     print(preds[i])
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()
# # exit()
# #
# # thresh = 0.95
# # print('#predictions<', str(thresh), 'for true examples: ', sum(preds[:,1]<thresh))
# # print('out of ', preds.shape[0], 'true examples', '- percentage:', sum(preds[:,1]<thresh)/(preds.shape[0]))
# # print(np.nonzero(preds[:,1]<thresh))
# #
# # #exit()
# #
# # false_indices = (y_val[:100,0]==1)
# # false_x = X_val[:100][false_indices]
# # preds = network.model.predict(false_x)
# # print('#predictions<', str(thresh), 'for false examples: ', sum(preds[:,0]<thresh))
# # print('out of ', preds.shape[0], 'false examples', '- percentage:', sum(preds[:,0]<thresh)/(preds.shape[0]))

# cwd = 'C:/Users/Rani/Desktop/Deep Project/project/'
# dataset_path = cwd + 'image_dataset/'
#
# X_train = np.load(dataset_path+'X_train.npy')
# X_val = np.load(dataset_path+'X_val.npy')
# y_train = np.load(dataset_path+'y_train.npy')
# y_val = np.load(dataset_path+'y_val.npy')
#
# network = Network()
# network.fit(X_train, y_train, X_val, y_val)
# network.model.save_weights('weights_3(new).h5')
