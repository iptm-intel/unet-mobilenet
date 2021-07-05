from tensorflow.keras.models import load_model
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
net = load_model('./models/val_42.h5', compile = False)
img = cv2.imread('./adas/route1/img/16531_2ddf1_57c6e3aebf30f.jpg', cv2.IMREAD_COLOR)
img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
img = img.astype(np.float32) / 255.
img = np.expand_dims(img, 0)
out = model.predict(img)
out = out[0]
classes = cv2.split(out)
class1 = classes[0]
class2 = classes[1]
plt.subplot(1, 2, 1)
plt.imshow(class1)
plt.subplot(1, 2, 2)
plt.imshow(class2)
plt.show()
