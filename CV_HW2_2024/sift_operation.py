import cv2 as cv
import os
import numpy as np
import random
import math
from matplotlib import pyplot as plt
from tqdm import tqdm

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
    
    def knn_find_good_match(self, threshold=0.75):
        # KNN
        print(len(self.keypoint1))
        print("KNN matching ...")
        k_matches = []
        for i,d1 in enumerate( self.descriptor1 ):
            print(i)
            min_kp = [-1,np.inf]
            sec_min_kp = [-1,np.inf]
            for j,d2 in enumerate( self.descriptor2 ):
                dist = np.linalg.norm(d1 - d2)
                if min_kp[1] > dist:
                    sec_min_kp = np.copy(min_kp)
                    min_kp = [j,dist]
                elif sec_min_kp[1] > dist:
                    sec_min_kp = [j,dist]
            k_matches.append((min_kp,sec_min_kp))
        # ratio test
        print("Ratio test ...")
        matches = []
        for i,(m1,m2) in enumerate(k_matches):
            # print("index : {}".format(i),m1,m2)
            # print(m1[1] , threshold*m2[1])
            if m1[1] < threshold*m2[1]:
                # unpacking the tuple to let one match stores 4 element (p1.x , p1.y , p2.x , p2.y)  
                # It doesn't mean summoing up two position
                matches.append(list( self.keypoint1[i].pt + self.keypoint2[m1[0]].pt))
        return np.array(matches)
    
    def find_homography(self, pairs):
        A = np.zeros((8,9))
        array_a = 0
        array_b = 4
        for i in range(pairs.shape[0]):
            
            x1   = pairs[i][0]
            y1   = pairs[i][1]
            x1_b = pairs[i][2]
            y1_b = pairs[i][3]

            A[array_a][0] = x1 
            A[array_a][1] = y1
            A[array_a][2] = 1
            A[array_a][6] = -x1*x1_b
            A[array_a][7] = -y1*x1_b
            A[array_a][8] = -x1_b
            
            A[array_b][3] = x1 
            A[array_b][4] = y1
            A[array_b][5] = 1
            A[array_b][6] = -x1*y1_b
            A[array_b][7] = -y1*y1_b
            A[array_b][8] = -y1_b

            array_a += 1
            array_b += 1

        U, s, V = np.linalg.svd(A)   # h = the eigenvector of ATA associated with the smallest eigenvalue -> the last one
        H = V[-1].reshape(3, 3)
        H = H/H[2, 2]     # standardize

        return H
    
    def find_error(self, points, H):
        num_points = points.shape[0]
        all_p1 = np.column_stack((points[:, :2], np.ones(num_points)))
        all_p2 = points[:, 2:4]
        
        temp = np.dot(H, all_p1.T)
        estimate_p2 = (temp[:2] / temp[2]).T
        
        errors = np.linalg.norm(all_p2 - estimate_p2, axis=1) ** 2

        return errors
    
    def ransac_find_best_H(self, good_matches, ransac_threshold=5, iteration=1000):
        num_best_inliers = 0
        for i in range( iteration ):
            # get random for pairs
            pairs = np.array( [ good_matches[idx] for idx in random.sample(range(len(good_matches)), 4) ] )
            H = self.find_homography(pairs)

            errors = self.find_error(good_matches, H)
            inliers = good_matches[ np.where(errors < ransac_threshold)[0] ]

            num_inliers = len(inliers)
            if num_inliers > num_best_inliers:
                best_inliers = inliers.copy()
                num_best_inliers = num_inliers
                best_H = H.copy()
        
        print("inliers/matches: {}/{}".format(num_best_inliers, len(good_matches)))

        return best_inliers, best_H


    
    def frame_match( self, MY_FUNCTION=False ):
        if MY_FUNCTION == False:
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
        
        else:
            good_match = self.knn_find_good_match()
            print(len(good_match))
            _, H = self.ransac_find_best_H(good_match)

            return H