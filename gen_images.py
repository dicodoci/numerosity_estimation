# Code based on generate_synthetic_images.py by Tom Runia
# https://gist.github.com/tomrunia/815ebd15dbaf02f60c83735061092f62

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import random
import math
import os

cv_FILLED = -1 #instead of using cv2.FILLED

def check_no_overlap(image, x, y, height, width):
    h, w = image.shape[0], image.shape[1]
    for k in range(-1, height+1):
        for l in range(-1, width+1):
            if y+k>=h or x+l>=w or y+k<0 or x+l<0:
                return False
            if image[y+k][x+l] != 0:
                return False
    return True

def draw_square(image, height, width):
    h, w = image.shape[0], image.shape[1]
    options = []
    for i in range(h):
        for j in range(w):
            if check_no_overlap(image, j, i, height, width):
                options.append((j,i))
    if options == []:
        return None
    (start_x, start_y) = options[np.random.randint(len(options))]
    p1 = (start_x, start_y)
    p2 = (start_x + width, start_y + height)
    cv2.rectangle(image, p1, p2, 255, cv_FILLED, cv2.CV_AA)
    return image


def generate_example(im_size=(30,30), min_obj=1, max_obj=32, sum_surfaces=[32, 64, 96, 128, 160, 192, 224, 256]):
    image = np.zeros((im_size[0],im_size[1]), dtype=np.uint8)
    num_obj = np.random.randint(min_obj, max_obj+1)
    if len(sum_surfaces) > 1:
        sum_surface = np.random.choice(sum_surfaces)
    else:
        sum_surface = sum_surfaces[0]
    for num_obj_todo in range(num_obj, 0, -1):
        avg_surface = sum_surface/num_obj + np.random.normal(0, 0.15)
        if avg_surface>0:
            width = int(round(math.sqrt(avg_surface) + np.random.normal(0, 0.3)))
            height = int(round(math.sqrt(avg_surface) + np.random.normal(0, 0.3)))
        else:
            width = 1
            height = 1
        if width < 1 :
            width = 1
        if height < 1 :
            heigth = 1
        image = draw_square(image, height, width)
        if image is None:
            return generate_example(im_size=im_size, min_obj=min_obj, max_obj=max_obj, sum_surfaces=sum_surfaces)
        sum_surface -= width*height
    return image



if __name__ == "__main__":

    for sum_surface in [32, 64, 96, 128, 160, 192, 224, 256]:
        for num_obj in range(1, 33):
            directory = "generated_images/surf" + str(sum_surface) + "_obj"+ str(num_obj)
            print(directory)
            if not os.path.exists(directory):
                os.makedirs(directory)
            for i in range(200):
                image = generate_example(min_obj=num_obj, max_obj=num_obj, sum_surfaces=[sum_surface])
                cv2.imwrite(directory +"/image_"+str(sum_surface)+"_"+str(num_obj)+"_"+str(i)+".png", image)
