import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, concatenate
from keras.models import Model
from keras.optimizers import adam


def unet():
    #Encoding
    inputs = Input(shape = (256,256,1))
    conv1_1 = Conv2D(64, (3,3), strides = 1, activation = 'relu', padding = 'same')(inputs)
    conv1_2 = Conv2D(64, (3,3), strides = 1, activation = 'relu', padding = 'same')(conv1_1)
    pool1 = MaxPooling2D(pool_size = (2,2))(conv1_2)
    conv2_1 = Conv2D(128, (3,3), strides = 1, activation = 'relu', padding = 'same')(pool1)
    conv2_2 = Conv2D(128, (3,3), strides = 1, activation = 'relu', padding = 'same')(conv2_1)
    pool2 = MaxPooling2D(pool_size = (2,2))(conv2_2)
    conv3_1 = Conv2D(256, (3,3), strides = 1, activation = 'relu', padding = 'same')(pool2)
    conv3_2 = Conv2D(256, (3,3), strides = 1, activation = 'relu', padding = 'same')(conv3_1)
    pool3 = MaxPooling2D(pool_size = (2,2))(conv3_2)
    conv4_1 = Conv2D(512, (3,3), strides = 1, activation = 'relu', padding = 'same')(pool3)
    conv4_2 = Conv2D(512, (3,3), strides = 1, activation = 'relu', padding = 'same')(conv4_1)
    pool4 = MaxPooling2D(pool_size = (2,2))(conv4_2)
    conv5_1 = Conv2D(1024, (3,3), strides = 1, activation = 'relu', padding = 'same')(pool4)
    conv5_2 = Conv2D(1024, (3,3), strides = 1, activation = 'relu', padding = 'same')(conv5_1)
    #Decoding
    upsample1 = UpSampling2D(size = (2,2), interpolation = 'nearest')(conv5_2)
    concat1 = concatenate([conv4_2, upsample1], axis = 3)
    conv6_1 = Conv2D(512, (3,3), strides = 1, padding = 'same', activation = 'relu')(concat1)
    conv6_2 = Conv2D(512, (3,3), strides = 1, padding = 'same', activation = 'relu')(conv6_1)
    upsample2 = UpSampling2D(size = (2,2), interpolation = 'nearest')(conv6_2)
    concat2 = concatenate([conv3_2, upsample2], axis = 3)
    conv7_1 = Conv2D(256, (3,3), strides = 1, padding = 'same', activation = 'relu')(concat2)
    conv7_2 = Conv2D(256, (3,3), strides = 1, padding = 'same', activation = 'relu')(conv7_1)
    upsample3 = UpSampling2D(size = (2,2), interpolation = 'nearest')(conv7_2)
    concat3 = concatenate([conv2_2, upsample3], axis = 3)
    conv8_1 = Conv2D(128, (3,3), strides = 1, padding = 'same', activation = 'relu')(concat3)
    conv8_2 = Conv2D(128, (3,3), strides = 1, padding = 'same', activation = 'relu')(conv8_1)
    upsample4 = UpSampling2D(size = (2,2), interpolation = 'nearest')(conv8_2)
    concat4 = concatenate([conv1_2, upsample4], axis = 3)
    conv9_1 = Conv2D(64, (3,3), strides = 1, padding = 'same', activation = 'relu')(concat4)
    conv9_2 = Conv2D(64, (3,3), strides = 1, padding = 'same', activation = 'relu')(conv9_1)
    conv9_3 = Conv2D(2, (1,1), strides = 1, padding = 'same', activation = 'relu')(conv9_2)
    conv10 = Conv2D(1, (1,1), strides = 1, padding = 'same', activation = 'sigmoid')(conv9_3)

    model = Model(input = inputs, output = conv10)
    model.compile(optimizer = adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

model = unet()
print(model.summary())