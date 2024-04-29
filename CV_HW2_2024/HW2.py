import cv2 as cv
import numpy as np
import random
import math
import sys
import os
from tqdm import tqdm

import feature_match as fm

file_path = "/home/wei/computer_vision_2024_spring/CV_HW2_2024/Photos/Base"

# read the image file & output the color & gray image
def read_img( path ):
    # opencv read image in BGR color space
    img = cv.imread( path )
    img_gray = cv.cvtColor( img, cv.COLOR_BGR2GRAY )

    return img, img_gray

# the dtype of img must be "uint8" to avoid the error of SIFT detector
def img_to_gray( img ):
    if img.dtype != "uint8":
        print( "The input image dtype is not uint8 , image type is : ",img.dtype )
        return
    img_gray = cv.cvtColor( img, cv.COLOR_BGR2GRAY )
    return img_gray

# create a window to show the image
# It will show all the windows after you call im_show()
# Remember to call im_show() in the end of main
def creat_im_window(window_name,img):
    cv.imshow(window_name,img)

# show the all window you call before im_show()
# and press any key to close all windows
def im_show():
    cv.waitKey(0)
    cv.destroyAllWindows()

def find_sift_kp_and_des( img_gray ):
    SIFT_Descriptor = cv.SIFT_create()
    kp, des = SIFT_Descriptor.detectAndCompute( img_gray, None )

    return kp, des

def stitch_img(left, right, H):
    print("stiching image ...")
    
    # Convert to double and normalize. Avoid noise.
    left = cv.normalize(left.astype('float'), None, 
                            0.0, 1.0, cv.NORM_MINMAX)   
    # Convert to double and normalize.
    right = cv.normalize(right.astype('float'), None, 
                            0.0, 1.0, cv.NORM_MINMAX)   
    
    # left image
    height_l, width_l, channel_l = left.shape
    corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]
    corners_new = [np.dot(H, corner) for corner in corners]
    corners_new = np.array(corners_new).T 
    x_news = corners_new[0] / corners_new[2]
    y_news = corners_new[1] / corners_new[2]
    y_min = min(y_news)
    x_min = min(x_news)

    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H = np.dot(translation_mat, H)
    
    # Get height, width
    height_new = int(round(abs(y_min) + height_l))
    width_new = int(round(abs(x_min) + width_l))
    size = (width_new, height_new)

    # right image
    warped_l = cv.warpPerspective(src=left, M=H, dsize=size)

    height_r, width_r, channel_r = right.shape
    
    height_new = int(round(abs(y_min) + height_r))
    width_new = int(round(abs(x_min) + width_r))
    size = (width_new, height_new)
    

    warped_r = cv.warpPerspective(src=right, M=translation_mat, dsize=size)
     
    black = np.zeros(3)  # Black pixel.
    
    # Stitching procedure, store results in warped_l.
    for i in tqdm(range(warped_r.shape[0])):
        for j in range(warped_r.shape[1]):
            pixel_l = warped_l[i, j, :]
            pixel_r = warped_r[i, j, :]
            
            if not np.array_equal(pixel_l, black) and np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_l
            elif np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_r
            elif not np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = (pixel_l + pixel_r) / 2
            else:
                pass
                  
    stitch_image = warped_l[:warped_r.shape[0], :warped_r.shape[1], :]
    return stitch_image

if __name__ == '__main__':
    if not os.path.exists(file_path):
        print("The file path does not exist.")
        exit()

    img_file = os.listdir(file_path)

    img = cv.imread( os.path.join(file_path, img_file[0].split('.')[0] + ".jpg") )

    output_img_size = (img.shape[1], img.shape[0])

    for i in range(len(img_file) - 1):
        now_img_path = os.path.join(file_path, img_file[i].split('.')[0] + ".jpg")
        next_img_path = os.path.join(file_path, img_file[i + 1].split('.')[0] + ".jpg")

        now_img, now_gray = read_img(now_img_path)
        next_img, next_gray = read_img(next_img_path)

        now_kp, now_des = find_sift_kp_and_des(now_gray)
        print("now kp:", len(now_kp))

        next_kp, next_des = find_sift_kp_and_des(next_gray)
        print("next kp:", len(next_kp))

        matcher = fm.feature_match(now_img, next_img, now_kp, next_kp, now_des, next_des)
        Homography_matrix = matcher.frame_match()

        stitched_img = stitch_img(now_img, next_img, Homography_matrix)

        cv.imshow("Stitched Image", stitched_img)
        cv.waitKey(0)
        cv.destroyAllWindows()