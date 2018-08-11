from keras.utils.np_utils import to_categorical
import pandas as pd
import numpy as np
import random
import sys

from image_augmentor import augmentor

# fer2013 dataset:
# Training       28709
# PrivateTest     3589
# PublicTest      3589

# emotion labels from FER2013:
emotion = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,
           'Sad': 4, 'Surprise': 5, 'Neutral': 6}
emo     = ['Angry', 'Fear', 'Happy',
           'Sad', 'Surprise', 'Neutral']

def reconstruct(pix_str, size=(48,48)):                                     #pixel value series => 48*48 image
    pix_arr = np.array(list(map(int, pix_str.split())))
    return pix_arr.reshape(size)

def emotion_count(y_train, classes, verbose=True):
    emo_classcount = {}
    print("Disgust classified as Angry")
    y_train.loc[y_train == 1] = 0
    classes.remove('Disgust')
    for new_num, _class in enumerate(classes):
        y_train.loc[(y_train == emotion[_class])] = new_num
        class_count = sum(y_train == (new_num))
        if verbose:
            print('{}: {} with {} samples'.format(new_num, _class, class_count))
        emo_classcount[_class] = (new_num, class_count)
    return y_train.values, emo_classcount
    
"""
def data_augmentation_by_flip(X_train,y_train):
    fliped_X = X_train[:,:,:,::-1]
    fliped_y = y_train
    X_train = np.concatenate((X_train, fliped_X))
    y_train = np.concatenate((y_train, fliped_y))
    return X_train, y_train
"""

def load_data(sample_split=0.3, usage='Training', to_cat=True, verbose=True, augmentation=False,
              classes=['Angry','Happy'], filepath='../../data/FER2013/fer2013.csv'):
    df = pd.read_csv(filepath)
    # print df.tail()
    # print df.Usage.value_counts()
    if usage == 'Training':
        df = df[df.Usage == 'Training']
    if usage == 'Validation':
        df = df[df.Usage == 'PrivateTest']
    elif usage == 'Test':
        df = df[df.Usage == 'PublicTest'] 
        #df2 = df[df.Usage == 'PrivateTest']
        #df = pd.concat([df1, df2], ignore_index=True)
    frames = []
    classes.append('Disgust')
    for _class in classes:
        class_df = df[df['emotion'] == emotion[_class]]
        frames.append(class_df)
    data = pd.concat(frames, axis=0)
    rows = random.sample(list(data.index), int(len(data)*sample_split))
    data = data.ix[rows]
    print('{} set for {}: {}'.format(usage, classes, data.shape))
    data['pixels'] = data.pixels.apply(lambda x: reconstruct(x))
    x = np.array([mat for mat in data.pixels]) # (n_samples, img_width, img_height)
    X_train = x.reshape(-1, 1, x.shape[1], x.shape[2])
    y_train, new_dict = emotion_count(data.emotion, classes, verbose)
    print(new_dict)
    if to_cat:
        y_train = to_categorical(y_train)
    
    if augmentation:
        X_train, y_train = augmentor(X_train,y_train)
    return X_train, y_train, new_dict

def save_data(X_train, y_train, usage='Training', fname='', folder='../../data/FER2013/'):
    np.save(folder + 'X_' + usage + fname, X_train)
    np.save(folder + 'y_' + usage + fname, y_train)

if __name__ == '__main__':
    # makes the numpy arrays ready to use:
    print('Making moves...')
    emo = ['Angry', 'Fear', 'Happy',
           'Sad', 'Surprise', 'Neutral']
    X_train, y_train, emo_dict = load_data(sample_split=1.0,
                                           classes=emo,
                                           usage='Training',
                                           verbose=True,
                                           augmentation=True)
    print('Saving...')
    save_data(X_train, y_train, usage="Training", fname='6_5pct')
    X_valid, y_valid, emo_dict = load_data(sample_split=1.0,
                                           classes=emo,
                                           usage='Validation',
                                           verbose=True,
                                           augmentation=False)
    print('Saving...')
    save_data(X_valid, y_valid, usage="Validation", fname='6_5pct')
    X_test, y_test, emo_dict = load_data(sample_split=1.0,
                                           classes=emo,
                                           usage='Test',
                                           verbose=True,
                                           augmentation=False)
    save_data(X_test, y_test, usage="Test", fname='6_5pct')
    print('Saving...')
    print("X_train.shape :", X_train.shape)
    print("y_train.shape :", y_train.shape)
    print("X_valid.shape :", X_valid.shape)
    print("y_valid.shape :", y_valid.shape)
    print("X_test.shape :", X_test.shape)
    print("y_test.shape :", y_test.shape)
    print('Done!')
