import os
os.environ['SM_FRAMEWORK'] = 'tf.keras' # такую штуку над делать если используешь tensorflow.keras, а не просто керас
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.losses import binary_crossentropy, mse
import segmentation_models as sm
from segmentation_models.losses import dice_loss, jaccard_loss, bce_jaccard_loss
from albumentations import (
    Compose, OneOf, GaussNoise, GaussianBlur, MedianBlur, Transpose, ShiftScaleRotate,
    RandomRotate90, RandomBrightnessContrast, RandomGamma, IAAEmboss, IAASharpen,
    IAAAffine, Cutout, FancyPCA, ToGray, ToSepia, VerticalFlip, HorizontalFlip,
    RandomGridShuffle, CLAHE, MotionBlur
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
)
from um import make_net
from seg_gen import Generator

def strong_aug(p=0.5):
    return Compose([
        RandomGridShuffle((2, 2), p=0.75),
        OneOf([
            ShiftScaleRotate(shift_limit=0.125),
            Transpose(),
            RandomRotate90(),
            VerticalFlip(),
            HorizontalFlip(),
            IAAAffine(shear=0.1)
        ]),
        OneOf([
            GaussNoise(),
            GaussianBlur(),
            MedianBlur(),
            MotionBlur()
        ]),
        OneOf ([
            RandomBrightnessContrast(),
            CLAHE(),
            IAASharpen()
        ]),
        Cutout(10, 2, 2, 127),
    ], p=p)
model = make_net((256, 256, 3))
model.summary()
model.compile(Nadam(1e-4), jaccard_loss, [jaccard_loss, 'binary_accuracy', 'mae'])
train_gen = Generator((256, 256, 3), 2, 'seg_anns_copy.csv', './images', './masks', strong_aug())
test_gen = Generator((256, 256, 3), 1, 'seg_anns_copy.csv', './images', './masks') # на валидации лучше ставить батч 1, с большем батчем результаты будут завышены
callbacks = [
    ModelCheckpoint(
        './models/val_{epoch}.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1,
        mode='min'
    )
]
model.fit(train_gen, epochs=100, verbose=1, validation_data=test_gen, callbacks=callbacks)
    
