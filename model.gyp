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
    convtrans1_1 = Conv2DTranspose(512, (3,3), strides = 1, padding = 'same', activation = 'relu')(concat1)
    convtrans1_2 = Conv2DTranspose(512, (3,3), strides = 1, padding = 'same', activation = 'relu')(convtrans1_1)
    upsample2 = UpSampling2D(size = (2,2), interpolation = 'nearest')(convtrans1_2)
    concat2 = concatenate([conv3_2, upsample2], axis = 3)
    convtrans2_1 = Conv2DTranspose(256, (3,3), strides = 1, padding = 'same', activation = 'relu')(concat2)
    convtrans2_2 = Conv2DTranspose(256, (3,3), strides = 1, padding = 'same', activation = 'relu')(convtrans2_1)
    upsample3 = UpSampling2D(size = (2,2), interpolation = 'nearest')(convtrans2_2)
    concat3 = concatenate([conv2_2, upsample3], axis = 3)
    convtrans3_1 = Conv2DTranspose(128, (3,3), strides = 1, padding = 'same', activation = 'relu')(concat3)
    convtrans3_2 = Conv2DTranspose(128, (3,3), strides = 1, padding = 'same', activation = 'relu')(convtrans3_1)
    upsample4 = UpSampling2D(size = (2,2), interpolation = 'nearest')(convtrans3_2)
    concat4 = concatenate([conv1_2, upsample4], axis = 3)
    convtrans4_1 = Conv2DTranspose(64, (3,3), strides = 1, padding = 'same', activation = 'relu')(concat4)
    convtrans4_2 = Conv2DTranspose(64, (3,3), strides = 1, padding = 'same', activation = 'relu')(convtrans4_1)
    convtrans4_3 = Conv2DTranspose(2, (1,1), strides = 1, padding = 'same', activation = 'relu')(convtrans4_2)

    model = Model(input = inputs, output = convtrans4_3)
    model.compile(optimizer = adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

model = unet()
print(model.summary())