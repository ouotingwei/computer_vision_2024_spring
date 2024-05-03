import numpy as np
import cv2 as cv
import numpy as np
import random
import math
from matplotlib import pyplot as plt

class feature_match:
    def __init__(self, img1, img2, keypoint1, keypoint2, descriptor1, descriptor2):
        # input images
        self.img1 = cv.cvtColor(img1, cv.COLOR_RGB2BGR)
        self.img2 = cv.cvtColor(img2, cv.COLOR_RGB2BGR)

        # input keypoints
        self.keypoint1 = keypoint1
        self.keypoint2 = keypoint2

        # input descriptors
        self.descriptor1 = descriptor1
        self.descriptor2 = descriptor2

    def knn_matcher(self, k=2):
        pass

    def find_best_homography_matrix()

    def find_homography_matrix( src_Point, dst_Point, ransac_threshold=4):
        A = []
        for i in range( len(src_Point) ):
            A.append( [src_Point[i, 0], src_Point[i, 1], 1, 0, 0, 0, -src_Point[i, 0], -dst_Point[i, 0], -src_Point[i, 1] * dst_Point[i ,0], -dst_Point[i, 0] ] )
            A.append([0, 0, 0, src_Point[i, 0], src_Point[i, 1], 1, -src_Point[i, 0] * dst_Point[i, 1], -src_Point[i, 1] * dst_Point[i, 1], -dst_Point[i, 1]])

        # solve linear eq Ah = 0 (SVD)
        u, sigma, vt = np.linalg.svd(A)

        H = np.reshape( vt[8], (3, 3) )
        H = ( 1/H.item(8) * H

        iteration = 2000
        max_inliner_num = 0
        BEST_H = None

        for iter in range(iteration):
            random_sample_idx = random.sample(range(len(matches)), 4)

        return BEST_H
    
    def frame_match(self):
        # Create BFMatcher object with cross check
        bf = cv.BFMatcher()
        
        # Match descriptors
        matches = bf.knnMatch(self.descriptor1,self.descriptor2, k=2)

        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append( (m.trainIdx, m.queryIdx) )
        
        
        print("There are ", len(good), 'Points with good match')

        if len(good) > 4:
            src_Point = np.float32( [self.keypoint1[j].pt for (_, j) in good] )
            dst_Point = np.float32( [self.keypoint2[i].pt for (i, _) in good] )

            print(src_Point[0])

            H, status = cv.findHomography(src_Point, dst_Point, cv.RANSAC, ransacReprojThreshold=4.0)

            return H

        else:
            print("Input imgs is not overlapping with each other")
            return None

        img3=np.empty((0, 0))
        img3 = cv.drawMatchesKnn(self.img1,self.keypoint1,self.img2,self.keypoint2,good,img3,flags=2) 
        plt.imshow(img3),plt.show()

