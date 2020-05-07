# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 21:09:33 2020

@author: burwe
"""

import pandas as pd
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

data = pd.read_csv('sudoku_dataset-master//outlines_sorted.csv')
dataset = []
target = []

qscale = 360

for one in range(data.shape[0]):
    line = data.iloc[one]
    img = cv2.imread('sudoku_dataset-master//' + line.filepath[1:])
    seq = line.values[1:].reshape(4, 2).astype('float32')
    seq[2], seq[3] = [line.p4_x, line.p4_y], [line.p3_x, line.p3_y]

    pts3 = seq
    pts4 = np.float32([[0,0],[qscale,0],[0,qscale],[qscale,qscale]])

    M = cv2.getPerspectiveTransform(pts3,pts4)
    dst = cv2.warpPerspective(img,M,(qscale,qscale))
    
    
    
    dst_im = Image.fromarray(dst)
    array = np.array(dst_im)[:,:,0]
    array = 255 - array
    divisor = array.shape[0] // 9
    puzzle = []
    for x in range(9):
        row = []
        for y in range(9):
            part = dst[x*divisor:(x+1)*divisor, y*divisor:(y+1)*divisor]
            part = part[divisor//10:-divisor//10, divisor//10:-divisor//10,]
            row.append(part)
        puzzle.append(row)
    dataset.append(puzzle)

    with open('sudoku_dataset-master//'+line.filepath[:-4] + '.dat') as f:
        text = f.read().rstrip().split('\n')[2:]
    
    digits = [int(j) for i in text for j in i.split()]
    target.append(digits)
    
dataset = np.array(dataset).reshape(202*9*9, int(divisor*0.8), int(divisor*0.8), 3) / 255
target = np.array(target).reshape(202*9*9)

digits = np.array([b for b in  target if b != 0])
dataset_with_digits = np.array([a for a,b in zip(dataset, target) if b != 0])

empty_spot = np.array([1 if i==0 else 0 for i in target])

from keras.utils import to_categorical
from keras import models, layers
from sklearn.model_selection import train_test_split

#"""
# create model which make difference between empty and non-empty cells

X_train, X_test, y_train, y_test = train_test_split(dataset, 
                                                    empty_spot,
                                                    random_state = 17)
model_bin = models.Sequential()

model_bin.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model_bin.add(layers.MaxPool2D(2, 2))
model_bin.add(layers.Conv2D(64, (3, 3), input_shape=(32, 32, 3)))
model_bin.add(layers.MaxPool2D(2, 2))
model_bin.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model_bin.add(layers.MaxPool2D(2, 2))

model_bin.add(layers.Flatten())

model_bin.add(layers.Dense(1024, activation='relu'))
model_bin.add(layers.Dropout(0.1))
model_bin.add(layers.Dense(256, activation='relu'))
model_bin.add(layers.Dropout(0.05))
model_bin.add(layers.Dense(96, activation='relu'))
model_bin.add(layers.Dropout(0.02))
model_bin.add(layers.Dense(32, activation='relu'))
model_bin.add(layers.Dropout(0.01))
model_bin.add(layers.Dense(16, activation='relu'))
model_bin.add(layers.Dense(units=1, activation='sigmoid'))

model_bin.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model_bin.fit(X_train, y_train, epochs=10,
          validation_data= [X_test, y_test])
# saved in pickle as model_bin

#"""

#'''
#create digit classification model

X_train, X_test, y_train, y_test = train_test_split(dataset_with_digits, 
                                                    to_categorical(digits)[:,1:],
                                                    random_state = 17)

model_ocr = models.Sequential()

model_ocr.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model_ocr.add(layers.MaxPool2D(2, 2))
model_ocr.add(layers.Conv2D(64, (3, 3), input_shape=(32, 32, 3)))
model_ocr.add(layers.MaxPool2D(2, 2))
model_ocr.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model_ocr.add(layers.MaxPool2D(2, 2))

model_ocr.add(layers.Flatten())

model_ocr.add(layers.Dense(1024, activation='relu'))
model_ocr.add(layers.Dropout(0.1))
model_ocr.add(layers.Dense(256, activation='relu'))
model_ocr.add(layers.Dropout(0.05))
model_ocr.add(layers.Dense(96, activation='relu'))
model_ocr.add(layers.Dropout(0.02))
model_ocr.add(layers.Dense(32, activation='relu'))
model_ocr.add(layers.Dropout(0.01))
model_ocr.add(layers.Dense(units=9, activation='softmax'))

model_ocr.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model_ocr.fit(X_train, y_train, epochs=15,
          validation_data= [X_test, y_test])

# saved in pickle as model_ocr

#'''

