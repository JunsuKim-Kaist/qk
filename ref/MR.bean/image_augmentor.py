import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

def augmentor(X_train,y_train,seed=1):
    
    X_swap = np.swapaxes(X_train,1,3)
    X_swap = np.swapaxes(X_swap,1,2)

    ia.seed(seed)

    seq = iaa.Sequential([
        #iaa.Flipud(0.5), # horizontally flip 50% of the images
         iaa.Affine(
             scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
             rotate=(-45, 45), # rotate by -45 to +45 degrees
             order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
         
            ),
        #iaa.AdditiveGaussianNoise(scale=0.2*255),
    ])

    images_aug = seq.augment_images(X_swap)
    
    X_train_aug = np.swapaxes(images_aug,1,3)
    X_train_aug = np.swapaxes(X_train_aug,2,3)
    
    X_train = np.concatenate((X_train, X_train_aug))
    y_train = np.concatenate((y_train, y_train))
    
    return X_train, y_train
