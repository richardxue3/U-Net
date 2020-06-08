from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt
from data import *
from model import *

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

    
trainingSet = trainGenerator(1,'data/membrane/train','image','label',aug_dict=data_gen_args, save_to_dir='data/membrane/traingenerator')
testingSet = testGenerator('data/membrane/test')
model = unet()
model.fit(trainingSet, steps_per_epoch = 500, epochs = 1)
results = model.predict_generator(testingSet,30,verbose=1)
saveResult("data/membrane/test",results)