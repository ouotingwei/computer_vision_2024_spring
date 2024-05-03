import cv2 as cv
import os
import numpy as np

import feature_match as fm
import stiching as stich

file_path = "/home/wei/computer_vision_2024_spring/CV_HW2_2024/Photos/Base"

# read the image file & output the color & gray image
def read_img( path ):
    # opencv read image in BGR color space
    img = cv.imread( path )
    img_gray = cv.cvtColor( img, cv.COLOR_BGR2GRAY )

    return cylinder_projection(img), cylinder_projection(img_gray)

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

def cylinder_projection( img, focal=3000 ):
    h, w = img.shape[:2]
    cylinder_projection_img = np.zeros_like( img )
    center_x, center_y = w//2, h//2

    for y in range( h ):
        for x in range( w ):
            theta = ( x - center_x ) / focal
            h_dash = ( y - center_y ) / np.cos( theta ) + center_y
            x_proj = int( np.sin(theta) * focal + center_x )
            y_proj = int( h_dash )
            if 0 <= x_proj < w and 0 <= y_proj < h:
                cylinder_projection_img[y, x] = img[y_proj, x_proj]
    
    return cylinder_projection_img

if __name__ == '__main__':
    if not os.path.exists(file_path):
        print("The file path does not exist.")
        exit()

    img_file = os.listdir(file_path)
    img_file.sort()
    print(img_file)

    img = []
    img_gray = []

    for i in range( len(img_file)-1 ):
        if i == 0:
            img_left_path = os.path.join(file_path, img_file[i].split('.')[0] + ".jpg")
        else:
            img_left_path = os.path.join("CV_HW2_2024/Photos/Base/result_image.jpg")

    img_1, img_1_gray = read_img(os.path.join(file_path, img_file[0].split('.')[0] + ".jpg"))

    img_2, img_2_gray = read_img(os.path.join(file_path, img_file[1].split('.')[0] + ".jpg"))
    img_3, img_3_gray = read_img(os.path.join(file_path, img_file[2].split('.')[0] + ".jpg"))

    img_1_kp, img_1_des = find_sift_kp_and_des(img_1_gray)
    img_2_kp, img_2_des = find_sift_kp_and_des(img_2_gray)
    img_3_kp, img_3_des = find_sift_kp_and_des(img_3_gray)

    matcher_1_2 = fm.feature_match(img_1, img_2, img_1_kp, img_2_kp, img_1_des, img_2_des)
    Homography_matrix_1_2 = matcher_1_2.frame_match()

    stitched_img_1_2 = stich.stitch_img(img_1, img_2, Homography_matrix_1_2)

    result_image_bgr = (stitched_img_1_2 * 255).astype(np.uint8)
    result_image_bgr = stich.removeBlackBorder(result_image_bgr)
    #result_image_bgr = cv.cvtColor(result_image_bgr, cv.COLOR_RGB2BGR)
    cv.imwrite('/home/wei/computer_vision_2024_spring/CV_HW2_2024/Photos/Base/result_image.jpg', result_image_bgr)

    img_1_2, img_1_2_gray = read_img("CV_HW2_2024/Photos/Base/result_image.jpg")
    img_1_2_kp, img_1_2_des = find_sift_kp_and_des(img_1_2_gray)

    matcher_1_2_3 = fm.feature_match(img_1_2, img_3, img_1_2_kp, img_3_kp, img_1_2_des, img_3_des)
    Homography_matrix_1_2_3 = matcher_1_2_3.frame_match()

    stitched_img_1_2_3 = stich.stitch_img(img_1_2, img_3, Homography_matrix_1_2_3)

    result_image_bgr = (stitched_img_1_2_3 * 255).astype(np.uint8)
    cv.imwrite('/home/wei/computer_vision_2024_spring/CV_HW2_2024/Photos/Base/result_image.jpg', result_image_bgr)
    
    #cv.imshow("Stitched Image", stitched_img)
    #cv.waitKey(0)
    #cv.destroyAllWindows()