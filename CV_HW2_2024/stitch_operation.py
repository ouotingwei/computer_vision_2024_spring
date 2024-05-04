import cv2 as cv
import numpy as np
from tqdm import tqdm

def stitch_img(left, right, H):
    
    # Convert to double and normalize. Avoid noise.
    left = cv.normalize(left.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)   
    # Convert to double and normalize.
    right = cv.normalize(right.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)   
    
    # left image
    height_l, width_l, channel_l = left.shape
    corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]
    corners_new = [np.dot(H, corner) for corner in corners]
    corners_new = np.array(corners_new).T        # [wx', wy', w]
    x_news = corners_new[0] / corners_new[2]     # x' = wx'/w
    y_news = corners_new[1] / corners_new[2]     # y' = wy'/w
    y_min = min(y_news)
    x_min = min(x_news)

    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H = np.dot(translation_mat, H)
    
    # Get height, width
    height_new = int(round(abs(y_min) + height_l))
    width_new = int(round(abs(x_min) + width_l))
    size = (width_new, height_new)

    warped_l = cv.warpPerspective(src=left, M=H, dsize=size)
    
    # right image
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
            
            if np.sum(pixel_l)>=np.sum(pixel_r):
                warped_l[i, j, :] = pixel_l
            else:
                warped_l[i, j, :] = pixel_r
                  
    stitch_image = warped_l[:warped_r.shape[0], :warped_r.shape[1], :]
    
    return stitch_image