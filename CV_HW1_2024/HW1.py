import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import scipy
from sklearn.preprocessing import normalize

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
    #smooth_img = cv2.GaussianBlur(image, (3, 3), 0)
    return image

def read_data(file_Dir):
    light_source = []
    cnt = 0
    with open( file_Dir + '/LightSource.txt', 'r' ) as file:
        for line in file.readlines():
            line = line.strip()
            position = list(map(int, line[line.find('(') + 1: line.find(')')].split(',')))
            light_source.append(np.array(position).astype(np.float32))
            cnt += 1

    # unit vector
    light_source = normalize(light_source, axis = 1)
    
    I_matrix = []

    for i in range(6):
        img_path = file_Dir + "/pic" + str(i+1) + ".bmp"
        img = read_bmp(img_path)
        I_matrix.append(img.ravel()) 

    I_matrix = np.asarray(I_matrix)

    return light_source, I_matrix

def find_normal(light_source, I_matrix):
    # least square solution
    KdN = np.linalg.solve(light_source.T @ light_source, light_source.T @ I_matrix).T

    # Normalize the normal vectors
    N = normalize(KdN, axis=0)

    # Visualize the normalized normal vectors
    normal_visualization(N)

    return N

def recover_surface( mask, N ):
    global image_row, image_col
    N = np.reshape( N, (image_row, image_col, 3) ) 
    num_pix_obj = np.size(np.where( mask!=0 )[0])

    # Solve Mz = V
    M = scipy.sparse.lil_matrix((2*num_pix_obj, num_pix_obj))
    v = np.zeros((2*num_pix_obj, 1))

    not_zero_row, not_zero_col = np.where( mask!=0 )

    full_object = np.zeros((image_row, image_col)).astype(np.int16)

    for cnt in range( num_pix_obj ):
        full_object[ not_zero_row[cnt], not_zero_col[cnt] ] = cnt
    
    for cnt in range( num_pix_obj ):
        row = not_zero_row[cnt]
        col = not_zero_col[cnt]

        n_x = N[row, col, 0]
        n_y = N[row, col, 1]
        n_z = N[row, col, 2]

        row_index = cnt * 2
        if mask[row, col+1] == True:  # right pixel exists!
            delta = full_object[row, col+1]
            M[row_index, cnt] = -1
            M[row_index, delta] = 1
            v[row_index] = -1 * n_x / n_z
        elif mask[row, col-1] == True:  # left pixel exists!
            delta = full_object[row, col-1]
            M[row_index, delta] = -1
            M[row_index, cnt] = 1
            v[row_index] = -1 * n_x / n_z

        row_index = cnt * 2 + 1
        if mask[row+1, col] == True:  # up pixel exists!
            delta = full_object[row+1, col]
            M[row_index, cnt] = 1
            M[row_index, delta] = -1
            v[row_index] = -1 * n_y / n_z
        elif mask[row-1, col] == True:  # down pixel exists!
            delta = full_object[row-1, col]
            M[row_index, delta] = 1
            M[row_index, cnt] = -1
            v[row_index] = -1 * n_y / n_z

    # find z from spsolve
    MTM = M.T @ M
    MTv = M.T @ v
    print(MTM)
    z = scipy.sparse.linalg.spsolve(MTM, MTv)

    # filter the outlier & optimize the surface
    depth = mask.astype(np.float32)

    normalized_z = (z - np.mean(z)) / np.std(z)
    outliner_idx = np.abs(normalized_z) > 10 # threshold for outlier
    z_max = np.max(z[~outliner_idx])
    z_min = np.min(z[~outliner_idx])

    for i in range(num_pix_obj):
        if z[i] > z_max:
            depth[not_zero_row[i], not_zero_col[i]] = z_max
        elif z[i] < z_min:
            depth[not_zero_row[i], not_zero_col[i]] = z_min
        else:
            depth[not_zero_row[i], not_zero_col[i]] = z[i]

    return depth

if __name__ == '__main__':
    file_Dir='/home/wei/CV_HW/CV_HW1_2024/test/star'
    light_source, I_matrix = read_data(file_Dir)
    N = find_normal( light_source, I_matrix)
    
    mask = read_bmp(file_Dir + '/pic1.bmp')
    threshold_value = 20
    max_value = 255
    ret, mask = cv2.threshold(mask, threshold_value, max_value, cv2.THRESH_BINARY)

    mask_visualization(mask)

    depth = recover_surface(mask, N)

    depth_visualization(depth)

    save_ply(depth, file_Dir + '/' + '1' + '.ply')
    show_ply(file_Dir + '/' + '1' + '.ply')

    # showing the windows of all visualization function
    plt.show()