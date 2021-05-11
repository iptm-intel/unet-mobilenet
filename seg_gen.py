import os
import csv
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import random
class Generator(Sequence):
    def __init__(self, img_shape, batch_size, ann_path, imgs_dir, masks_dir, augs=None):
        super(Generator, self).__init__()

        self.batch_size = batch_size
        self.img_shape = img_shape
        self.ann_path = ann_path
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.augs = augs
        self.anns = []
        with open(self.ann_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.anns.append(row)
        self.img_num = len(self.anns)
        self.indexes = list(range(self.img_num))
    def on_epoch_end(self):
        random.shuffle(self.indexes)
    def __len__(self):
        return int(len(self.anns) / self.batch_size)
    def __getitem__(self, index):
        batch_idxs = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_imgs = np.empty((self.batch_size, self.img_shape[0], self.img_shape[1], self.img_shape[2]), np.float32)
        batch_masks = np.empty((self.batch_size, self.img_shape[0], self.img_shape[1], 2), np.float32)
        for i, idx in enumerate(batch_idxs):
            ann = self.anns[idx]
            img = cv2.imread(os.path.join(self.imgs_dir, ann[0]))
            mask = np.load(os.path.join(self.masks_dir, ann[1]))
            if self.augs is not None:
                data = {'image':img, 'masks':cv2.split(mask)}
                augmented = self.augs(**data)
                img = augmented['image']
                mask = cv2.merge(augmented['masks'])
                mask[mask > 127] = 255
                mask[mask < 127] = 0
            img = img.astype(np.float32) / 255.
            mask = mask.astype(np.float32) / 255.
            batch_imgs[i] = img
            batch_masks[i] = mask
        return batch_imgs, batch_masks
if __name__ == '__main__':
    gen = Generator((256, 256, 3), 1, 'seg_anns_copy.csv', 'images', 'masks')
    for i in range(len(gen)):
        a, b = gen.__getitem__(i)
        masks = b[0, :, :, 0]
        masks = masks * 255.
        masks = masks.astype(np.uint8)
        cv2.imshow('test', masks)
        cv2.waitKey(30)
       
