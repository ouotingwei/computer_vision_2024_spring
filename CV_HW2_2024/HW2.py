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

    for i in range(1,  len(img_file)-1 ):
        if i == 0:
            img_left_path = os.path.join(file_path, img_file[0].split('.')[0] + ".jpg")
            img_right_path = os.path.join(file_path, img_file[1].split('.')[0] + ".jpg")
        else:
            img_left_path = os.path.join("CV_HW2_2024/Photos/Base/result_image.jpg")
            img_right_path = os.path.join(file_path, img_file[i+1].split('.')[0] + ".jpg")

        img_left, img_left_gray = read_img( img_left_path )
        img_right, img_right_gray = read_img( img_right_path )
    
        img_left_kp, img_left_des = find_sift_kp_and_des( img_left_gray )
        img_right_kp, img_right_des = find_sift_kp_and_des( img_right_gray )
    
        matcher = fm.feature_match(img_left, img_right, img_left_kp, img_right_kp, img_left_des, img_right_des)
        Homography_matrix = matcher.frame_match()
        stitched_img = stich.stitch_img(img_left, img_right, Homography_matrix)
    
        result_image_bgr = (stitched_img * 255).astype(np.uint8)
        result_image_bgr = stich.removeBlackBorder(result_image_bgr)

        # crop img into rectangle
        stitched = cv.copyMakeBorder(result_image_bgr, 10, 10, 10, 10, cv.BORDER_CONSTANT, (0, 0, 0))
        result_gray = cv.cvtColor( result_img_bgr, cv.COLOR_BGR2GRAY )
        threshhold = cv.trreshold( result_gray, 0, 255, cv.THRESH_BINARY)[1]
        cnts = cv.findCountours( threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIIMPLE)[0]

        mask = np.zeros( threshold.shape, dtype="uint8" )
        (x, y, w, h) = cv2.boundingRect( cnts[0] )
        cv.rectangle( mask, (x,y), (x+w,y+h ), 255, -1 )

        # erode until ok
        min_Rect = mask.copy()
        sub = mask.copy()
        while cv.countNonZero( sub ) > 0:
            minRect = cv2.erode( minRect, None )
            sub = cv.subtract( minRect, threshold )

        cnts = cv.findContours(minRect, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
        (x, y, w, h) = cv.boundingRect(cnts[0])
        stitched = stitched[y:y + h, x:x + w]
        
        cv.imwrite('/home/wei/computer_vision_2024_spring/CV_HW2_2024/Photos/Base/result_image.jpg', stitched)
    
    #cv.imshow("Stitched Image", stitched_img)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
