# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:33:41 2020

@author: burwe
"""

#general assembly

import cv2
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import decision_core as dc


def get_corners(path):
    # fit image for square and return corners
    
    im = cv2.imread(path)
    # processing image with filters
    proc = cv2.GaussianBlur(im.copy(), (9, 9), 0)
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    proc = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    proc = cv2.bitwise_not(proc, proc)
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
    proc = cv2.dilate(proc, kernel)

    # find contour of sudoku
    contours, _ = cv2.findContours(proc.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = contours[0]
    
    # get convex hull from polygon
    from scipy.spatial import ConvexHull
    points = np.array([i[0] for i in polygon])
    hull = ConvexHull(points)
    hull = hull.simplices
    
#cv2.drawContours(im, polygon, -1, (0, 0, 255), 10)

    frame = []
    for simplex in hull:
        #plt.plot(points[simplex, 0], points[simplex, 1], 'k-', color='brown')
        frame.append([points[simplex, 0], points[simplex, 1]])

    frame = np.array(frame)
    dots = np.array([i.T for i in frame])
    dots = dots.reshape(dots.shape[0]*2,2)
    
    min_x, max_x = dots[:, 0].min(), dots[:, 0].max()
    min_y, max_y = dots[:, 1].min(), dots[:, 1].max()
    left_bottom = min_x, min_y
    left_top = min_x, max_y
    right_top = max_x, max_y
    right_bottom = max_x, min_y
    
    corners = []
    for edge in (left_bottom, right_bottom, right_top, left_top):
        # choose dot with minimal euclidean scale 
        corner = np.array([np.linalg.norm( d - edge ) for d in dots]).argmin()
        corners.append(corner)
    #    cv2.circle(im, tuple(dots[corner]), 8, (255, 0, 0), -1)

    #plt.imshow(im)
    corners = [dots[i] for i in corners]
    return corners

# load list of images, you can change index 
data = pd.read_csv('sudoku_dataset-master//outlines_sorted.csv').iloc[15]
p = 'sudoku_dataset-master' + data.filepath
#p = 'photo_2020-05-05_23-15-40.jpg'
 
im = cv2.imread(p)

qscale = 360
seq = np.array(get_corners(p))
# wrong order of dots
seq[2] = get_corners(p)[3]
seq[3] = get_corners(p)[2]

pts1 = seq.astype('float32')
pts2 = np.float32([[0,0],[qscale,0],[0,qscale],[qscale,qscale]])

M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(im,M,(qscale,qscale))
dst_im = Image.fromarray(dst)

divisor = dst.shape[0] // 9

puzzle = []
predict_numbers = []

import pickle
model_bin = pickle.load(open('model_bin', 'rb'))
model_ocr = pickle.load(open('model_ocr', 'rb'))

for x in range(9):
    row = []
    for y in range(9):
        cell = dst[x*divisor:(x+1)*divisor, y*divisor:(y+1)*divisor]
        cell = cell[divisor//10:-divisor//10, divisor//10:-divisor//10,]
        row.append(cell)
        cell_relative = np.array(cell).reshape(1, 32, 32, 3) / 255
        if model_bin.predict(cell_relative) > 0.5: # check "is empty cell"
            predict_numbers.append(0)
        else:
            # predict number with early learned model
            predict_numbers.append(model_ocr.predict(cell_relative).argmax() + 1)
#    print(x)
    puzzle.append(row)

predict_numbers = np.array(predict_numbers).reshape(9, 9)
#solved_sudoku = dc.solveSudoku(predict_numbers)

plt.imshow(dst_im)
print('Найдены числа в судоку')
print(predict_numbers)
if dc.solveSudoku(predict_numbers):
    print('Судоку решен')
    print(predict_numbers)
