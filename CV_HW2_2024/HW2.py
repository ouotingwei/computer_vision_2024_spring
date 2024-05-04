import cv2 as cv
import os
import numpy as np

import sift_operation as si_o
import stitch_operation as st_o

file_path = "/home/wei/computer_vision_2024_spring/CV_HW2_2024/Photos/Base"
    
def removeBlackBorder( img ):
    return cv.medianBlur(img, 5)

# read the image file & output the color & gray image
def read_img( path, cylinder=False ):
    # opencv read image in BGR color space
    img = cv.imread( path )
    img_gray = cv.cvtColor( img, cv.COLOR_BGR2GRAY )
    if cylinder == False:
        return img, img_gray
    
    else:
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

def cylinder_projection( img, focal=2200 ):
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

def crop_img( img, crop_ratio=1):
    height, width = img.shape[:2]

    crop_width = int(width * crop_ratio)
    crop_height = int(height * crop_ratio)

    start_x = width - crop_width
    start_y = height - crop_height

    cropped_img = img[start_y:height, start_x:width]

    return cropped_img


def calculate_average_brightness(img_list):
    total_brightness = 0
    
    for img_path in img_list:
        img = cv.imread(img_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        current_brightness = cv.mean(gray)[0]
        total_brightness += current_brightness
    
    avg_brightness = total_brightness / len(img_list)
    
    return avg_brightness

def auto_adjust_brightness(image, target_brightness):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    current_brightness = cv.mean(gray)[0]
    ratio = target_brightness / current_brightness
    adjusted_image = cv.convertScaleAbs(image, alpha=ratio, beta=0)
    
    return adjusted_image

if __name__ == '__main__':
    if not os.path.exists(file_path):
        print("The file path does not exist.")
        exit()
    
    img_file = os.listdir(file_path)
    img_file.sort()
    print(img_file)
    files_cnt = len(img_file)
    
    img_paths = [os.path.join(file_path, img.split('.')[0] + ".jpg") for img in img_file]
    
    # calculate the average brightness
    avg_brightness = calculate_average_brightness(img_paths)
    print("Average brightness:", avg_brightness)
    
    # adjust brightness
    img_left = None
    img_left_gray = None
    
    img_left_path = img_paths[0]
    img_left = cv.imread(img_left_path)
    img_left = auto_adjust_brightness(img_left, avg_brightness)
    img_left_gray = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
    
    for i in range(files_cnt - 1):
        img_right = None
        img_right_gray = None

        img_right_path = img_paths[i + 1]
        img_right = cv.imread(img_right_path)
        img_right = auto_adjust_brightness(img_right, avg_brightness)
        img_right_gray = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

        print("left:", img_left_path)
        print("right:", img_right_path)

        img_left_kp, img_left_des = find_sift_kp_and_des(img_left_gray)
        img_right_kp, img_right_des = find_sift_kp_and_des(img_right_gray)

        matcher = si_o.feature_match(img_left, img_right, img_left_kp, img_right_kp, img_left_des, img_right_des)
        Homography_matrix = matcher.frame_match(MY_FUNCTION=False)
        stitched_img = st_o.stitch_img(img_left, img_right, Homography_matrix)

        result_image_bgr = (stitched_img * 255).astype(np.uint8)
        result_image_bgr = removeBlackBorder(result_image_bgr)

        cropped_img = crop_img(result_image_bgr)

        cv.imwrite('/home/wei/computer_vision_2024_spring/CV_HW2_2024/Photos/Base/result_image.jpg', cropped_img)

        img_left = cropped_img
        img_left_gray = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)