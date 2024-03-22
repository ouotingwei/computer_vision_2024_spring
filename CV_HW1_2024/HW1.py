import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import math
image_row = 120
image_col = 120

# visualizing the mask (size : "image width" * "image height")
def mask_visualization(M):
    mask = np.copy(np.reshape(M, (image_row, image_col)))
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')

# visualizing the unit normal vector in RGB color space
# N is the normal map which contains the "unit normal vector" of all pixels (size : "image width" * "image height" * 3)
def normal_visualization(N):
    # converting the array shape to (w*h) * 3 , every row is a normal vetor of one pixel
    N_map = np.copy(np.reshape(N, (image_row, image_col, 3)))
    # Rescale to [0,1] float number
    N_map = (N_map + 1.0) / 2.0
    plt.figure()
    plt.imshow(N_map)
    plt.title('Normal map')

# visualizing the depth on 2D image
# D is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def depth_visualization(D):
    D_map = np.copy(np.reshape(D, (image_row,image_col)))
    # D = np.uint8(D)
    plt.figure()
    plt.imshow(D_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')

# convert depth map to point cloud and save it to ply file
# Z is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def save_ply(Z,filepath):
    Z_map = np.reshape(Z, (image_row,image_col)).copy()
    data = np.zeros((image_row*image_col,3),dtype=np.float32)
    # let all point float on a base plane 
    baseline_val = np.min(Z_map)
    Z_map[np.where(Z_map == 0)] = baseline_val
    for i in range(image_row):
        for j in range(image_col):
            idx = i * image_col + j
            data[idx][0] = j
            data[idx][1] = i
            data[idx][2] = Z_map[image_row - 1 - i][j]
    # output to ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(filepath, pcd,write_ascii=True)

# show the result of saved ply file
def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])

# read the .bmp file
def read_bmp(filepath):
    global image_row
    global image_col
    image = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    image_row , image_col = image.shape
    # use gaussian blur
    smooth_img = cv2.GaussianBlur(image, (3, 3), 0)
    return smooth_img

def get_light_source_from_txt( file_Dir ):
    with open( file_Dir + '/LightSource.txt', 'r' ) as file:
        content = file.read()
        lines = content.strip().split("\n")
        light_source = np.zeros( (len(lines), 3), dtype=float)

        for i, line in enumerate( lines ):
            parts = line.strip().split(':')
            values = parts[1].strip()[1:-1].split(',')
            light_source[i] = [int(value) for value in values ]
    
    #print( light_source[0] ) 
    return light_source

def normalize_matrix( matrix ):
    norm = np.linalg.norm( matrix )
    normalize_matrix = matrix / norm
    return normalize_matrix

def estimate_surface_height(row, col, z_appro, normal):
    if normal[2] == 0:
        return np.nan

    surface_height = -((normal[0] / normal[2]) * col + (normal[1] / normal[2]) * row - z_appro)
    
    return surface_height

def recover_surface( file_Dir ):
    global image_row
    global image_col

    image = cv2.imread(file_Dir + "/pic1" + ".bmp",cv2.IMREAD_GRAYSCALE)
    image_row , image_col = image.shape

    I_matrix = np.empty((6, image_row * image_col))
    z_appro_map = np.zeros( (image_row, image_col) )

    
    for i in range(6):
        img_path = file_Dir + "/pic" + str(i+1) + ".bmp"
        img = read_bmp(img_path)

        img = img.reshape((-1, -1)).squeeze(1)
        I_matrix.append(img)

    light_source = get_light_source_from_txt(file_Dir)

    #KdN = (np.linalg.inv(light_source.T @ light_source) @ light_source.T) @ I_matrix
    KdN = np.linalg.lstsq(light_source, I_matrix, rcond=-1)[0].T

    normal_vector = KdN.T.reshape((image_row, image_col, 3))
    normal_visualization(normal_vector)

    normal_vector = normalize_matrix(normal_vector)

    # z-approximate
    for row in range(image_row):
        for col in range(image_col):
            z_appro_map[row][col] = ( -1 * normal_vector[row][col][0] / normal_vector[row][col][2]) * col + ( -1 * normal_vector[row][col][1] / normal_vector[row][col][2]) * row 
    
    # reconstruct the surface from the z-approximate
    surface_map = np.zeros((image_row, image_col))

    for row in range(image_row):
        for col in range(image_col):
            surface_map[row][col] = estimate_surface_height(row, col, z_appro_map[row][col], normal_vector[row][col])

    # smooth the surface
    depth_visualization(surface_map)   
    
    
if __name__ == '__main__':
    recover_surface( file_Dir='/home/tingweiou/computer_vision_2023_spring/CV_HW1_2024/test/star' )
    #depth_visualization(Z)
    #save_ply(Z,filepath)
    #show_ply(filepath)

    # showing the windows of all visualization function
    plt.show()