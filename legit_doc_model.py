# from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
from keras import regularizers


class Network_doc4x4:
    def __init__(self,input_shape=[225,425,1]):
        self.X = []
        self.y = []
        # self.num_classes = 100
        self.weight_decay = 0.0005
        self.x_shape = input_shape
        # self.x_shape = [300, 300, 1]  # i resized all examples to this size

        self.model = self.build_model()
        # self.model.load_weights('weights_doc_4x4_side_97.h5')
        # self.read_data()
        # self.pad_data_and_save()

    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

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

    def fit(self, X, y, X_val=None, y_val=None):
        # training parameters
        batch_size = 16
        maxepoches = 30
        learning_rate = 0.0001
        lr_decay = 1e-6
        lr_drop = 5

        def lr_scheduler(epoch):
            weights_file = 'weights_doc_epoch' + str(epoch) + '.h5'
            self.model.save_weights('weights_by_epoch/'+weights_file)
            return learning_rate * (0.5 ** (epoch // lr_drop))

        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        # optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        # self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model.fit(X, y, validation_data=(X_val, y_val), epochs=maxepoches,
                       batch_size=batch_size, shuffle=True, callbacks=[reduce_lr], verbose=2)  # verbose was 1
        # self.model.fit(X, y, validation_split=0.1, epochs = maxepoches,
        #               batch_size=batch_size, shuffle=True,callbacks=[reduce_lr], verbose=1)



# data_path = 'C:/Users/Rani/Desktop/Deep Project/project/document_dataset/'
# data_path += '5x5_side/'
# X_train = np.load(data_path+'X_train.npy')
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
# X_val = np.load(data_path+'X_val.npy')
# X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], X_val.shape[2], 1))
# y_train = np.load(data_path+'y_train.npy')
# y_val = np.load(data_path+'y_val.npy')
#
# input_size = list(X_val.shape)[1:]
# print(input_size)
#
# network = Network_doc4x4(input_shape=input_size)
# network.model.load_weights('weights_doc_5x5_side_94_3.h5')
# network.fit(X_train,y_train,X_val,y_val)
#
# network.model.save_weights('weights_by_epoch/weights_final_epoch.h5')


# network.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(network.model.evaluate(X_val, y_val))




#
# X_val_preds = network.model.predict(X_val[:1000])[:,1]
# print('X_val_preds.shape=', X_val_preds.shape)
#
# # false_positives = (X_val[:1000])[np.logical_and((X_val_preds>=0.5),y_val[:1000,1]==0)]
# # false_positives_predictions = X_val_preds[ np.logical_and((X_val_preds>=0.5),y_val[:1000,1]==0)]
# # print('# false positives=', false_positives.shape)
# # for i in range(len(false_positives)):
# #     cv2.imshow(str(i), false_positives[i])
# #     # print('actual label:',y_val[i], 'prediction:', X_val_preds[i])
# #     print('prediction:', false_positives_predictions[i])
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()
#
# false_negatives = (X_val[:1000])[np.logical_and((X_val_preds<=0.5),y_val[:1000,1]==1)]
# false_negatives_predictions = X_val_preds[np.logical_and((X_val_preds<=0.5),y_val[:1000,1]==1)]
# print('# false negatives=', false_negatives.shape)
# for i in range(len(false_negatives)):
#     cv2.imshow(str(i), false_negatives[i])
#     # print('actual label:',y_val[i], 'prediction:', X_val_preds[i])
#     print('prediction:', false_negatives_predictions[i])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# data_path = 'C:/Users/Rani/Desktop/Deep Project/project/document_dataset/'
# X_val = np.load(data_path+'X_val.npy')
# X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], X_val.shape[2], 1))
# y_val = np.load(data_path+'y_val.npy')
#
# input_size = list(X_val.shape)[1:]
#
# network = Network_doc4x4(input_shape=input_size)
# network.model.load_weights('weights_doc_new96.h5')
# network.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(network.model.evaluate(X_val, y_val))