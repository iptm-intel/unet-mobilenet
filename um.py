from tensorflow.keras.layers import (
    Input, Conv2D, DepthwiseConv2D, ReLU, Activation, concatenate, BatchNormalization, Conv2DTranspose, GlobalAveragePooling2D
    )
from tensorflow.keras.models import Model
import numpy as np

def convdws1(filters):
    def layer(x):
        convdw = DepthwiseConv2D(kernel_size=(3,3), padding="same")(x)
        convdw = BatchNormalization()(convdw)
        convdw = ReLU()(convdw)
        conv = Conv2D(filters=filters, kernel_size=(1,1), padding="same")(convdw)
        conv = BatchNormalization()(conv)
        conv = ReLU()(conv)
        return conv
    return layer

def convdws2(filters):
    def layer(x):
        convdw = DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding="same")(x)
        convdw = BatchNormalization()(convdw)
        convdw = ReLU()(convdw)
        conv = Conv2D(filters=filters, kernel_size=(1,1), padding="same")(convdw)
        conv = BatchNormalization()(conv)
        conv = ReLU()(conv)
        return conv
    return layer

def make_net(input_shape):
    img_layer = Input(shape = input_shape)
    conv1 = Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), padding="same")(img_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)
    down1 = convdws1(64)(conv1)
    down2 = convdws2(128)(down1)
    down3 = convdws1(128)(down2)
    down4 = convdws2(256)(down3)
    down5 = convdws1(256)(down4)
    down6 = convdws2(512)(down5)
    down7 = convdws1(512)(down6)
    down8 = convdws1(512)(down7)
    down9 = convdws1(512)(down8)
    down10 = convdws1(512)(down9)
    down11 = convdws1(512)(down10)
    up1 = Conv2DTranspose(filters=256, kernel_size=(3,3), strides=(2,2), padding="same")(down11)
    up1 = BatchNormalization()(up1)
    up1 = ReLU()(up1)
    up1 = concatenate([up1, down5])
    up1 = convdws1(256)(up1)
    up2 = Conv2DTranspose(filters=128, kernel_size=(3,3), strides=(2,2), padding="same")(up1)
    up2 = BatchNormalization()(up2)
    up2 = ReLU()(up2)
    up2 = concatenate([up2, down3])
    up2 = convdws1(128)(up2)
    up3 = Conv2DTranspose(filters=64, kernel_size=(3,3), strides=(2,2), padding="same")(up2)
    up3 = BatchNormalization()(up3)
    up3 = ReLU()(up3)
    up3 = concatenate([up3, down1])
    up3 = convdws1(64)(up3)
    up4 = Conv2DTranspose(filters=64, kernel_size=(3,3), strides=(2,2), padding="same")(up3)
    up4 = BatchNormalization()(up4)
    up4 = ReLU()(up4)
    conv_out = Conv2D(filters=2, kernel_size=(1,1), padding="same", activation="sigmoid")(up4)
    model = Model(inputs=[img_layer], outputs=[conv_out])
    return model

if __name__ == '__main__':
    net = make_net((224, 224, 3))
    net.compile('sgd', 'categorical_crossentropy')
    net.summary()
    test = np.zeros((1000, 224, 224, 3), np.float32)
    net.predict(test, 1, verbose=1)

