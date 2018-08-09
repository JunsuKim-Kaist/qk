from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
from log import save_model, save_config, save_result
from sklearn.cross_validation import train_test_split
import sklearn.metrics as metrics
import numpy as np
import time
import sys

def describe(X_shape, y_shape, batch_size, dropout, nb_epoch, conv_arch, dense):
    print(' X_train shape: ', X_shape) # (n_sample, 1, 48, 48)
    print(' y_train shape: ', y_shape) # (n_sample, n_categories)
    print('      img size: ', X_shape[2], X_shape[3])
    print('    batch size: ', batch_size)
    print('      nb_epoch: ', nb_epoch)
    print('       dropout: ', dropout)
    print('conv architect: ', conv_arch)
    print('neural network: ', dense)

def logging(model, starttime, batch_size, nb_epoch, conv_arch,dense, dropout,
            X_shape, y_shape, train_acc, val_acc, dirpath):
    now = time.ctime()
    model.save_weights('../../data/FER2013/weights/{}'.format(now))
    save_model(model.to_json(), now, dirpath)
    save_config(model.get_config(), now, dirpath)
    save_result(starttime, batch_size, nb_epoch, conv_arch, dense, dropout,
                    X_shape, y_shape, train_acc, val_acc, dirpath)

def cnn_architecture(X_train, y_train, conv_arch=[(32,3),(64,3),(128,3)],
                    dense=[64,2], dropout=0.5, batch_size=128, nb_epoch=100, validation_split=0.2, patience=5, dirpath='../../data/FER2013/results/'):
    starttime = time.time()
    X_train = X_train.astype('float32')
    X_shape = X_train.shape
    y_shape = y_train.shape
    describe(X_shape, y_shape, batch_size, dropout, nb_epoch, conv_arch, dense)

    # data augmentation:
    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=validation_split)
    # datagen = ImageDataGenerator(rescale=1./255,
    #                              rotation_range=10,
    #                              shear_range=0.2,
    #                              width_shift_range=0.2,
    #                              height_shift_range=0.2,
    #                              horizontal_flip=True)

    # datagen.fit(X_train)
    # model architecture:
    model = Sequential()
    model.add(Convolution2D(conv_arch[0][0], (3,3), padding='same', activation='relu',input_shape=(1, X_train.shape[2], X_train.shape[3])))

    if (conv_arch[0][1]-1) != 0:
        for i in range(conv_arch[0][1]-1):
            model.add(Convolution2D(conv_arch[0][0], (3,3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    if conv_arch[1][1] != 0:
        for i in range(conv_arch[1][1]):
            model.add(Convolution2D(conv_arch[1][0], (3,3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    if conv_arch[2][1] != 0:
        for i in range(conv_arch[2][1]):
            model.add(Convolution2D(conv_arch[2][0], (3,3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    model.add(Flatten())  # this converts 3D feature maps to 1D feature vectors
    if dense[1] != 0:
        for i in range(dense[1]):
            model.add(Dense(dense[0], activation='relu'))
            if dropout:
                model.add(Dropout(dropout))
    prediction = model.add(Dense(y_train.shape[1], activation='softmax'))

    # optimizer:
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # set callback:
    callbacks = []
    if patience != 0:
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
        callbacks.append(early_stopping)

    print('Training....')
    # fits the model on batches with real-time data augmentation:
    # hist = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
    #                 samples_per_epoch=len(X_train), nb_epoch=nb_epoch, validation_data=(X_test,y_test), callbacks=callbacks, verbose=1)

    '''without data augmentation'''
    hist = model.fit(X_train, y_train, epochs=nb_epoch, batch_size=batch_size,
              validation_split=validation_split, callbacks=callbacks, shuffle=True, verbose=1)

    # model result:
    train_val_accuracy = hist.history
    train_acc = train_val_accuracy['acc']
    val_acc = train_val_accuracy['val_acc']
    print('          Done!')
    print('     Train acc: ', train_acc[-1])
    print('Validation acc: ', val_acc[-1])
    print(' Overfit ratio: ', val_acc[-1]/train_acc[-1])

    logging(model, starttime, batch_size, nb_epoch, conv_arch, dense,
            dropout, X_shape, y_shape, train_acc, val_acc, dirpath)

    return model

def predict(model,X_test,y_test):
    predictions = model.predict(X_test)
    
    #print(predictions)
    pred = np.zeros_like(predictions)
    pred[np.arange(len(predictions)), predictions.argmax(1)] = 1
    
    acc = metrics.accuracy_score(y_test,pred)
    print("ACC metric in the test dataset", acc)
    return acc
    
if __name__ == '__main__':
    # import dataset:
    X_train_fname = '../../data/FER2013/X_train6_5pct.npy'
    y_train_fname = '../../data/FER2013/y_train6_5pct.npy'
    print('Loading data...')
    X_train = np.load(X_train_fname)
    y_train = np.load(y_train_fname)
    
    X_test_fname = '../../data/FER2013/X_test6_5pct.npy'
    y_test_fname = '../../data/FER2013/y_test6_5pct.npy'
    X_test = np.load(X_test_fname)
    y_test = np.load(y_test_fname)
    
    
    #model = cnn_architecture(X_train, y_train, conv_arch=[(32,3),(64,3),(128,3)], dense=[512,2], dropout=0.4, batch_size=128, nb_epoch=100, dirpath = '../../data/FER2013/results/')
    model = cnn_architecture(X_train, y_train, conv_arch=[(32,3),(64,3),(128,3)], dense=[64,2], batch_size=256, nb_epoch=1, dirpath = '../../data/FER2013/results/')
    test_accuracy = predict(model,X_test,y_test)

