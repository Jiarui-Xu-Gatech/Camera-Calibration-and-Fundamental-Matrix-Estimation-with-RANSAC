import numpy as np
import cv2 as cv
from vision.part3_ransac import ransac_fundamental_matrix


#import matplotlib.pyplot as plt

def panorama_stitch(imageA, imageB):
    """
    ImageA and ImageB will be an image pair that you choose to stitch together
    to create your panorama. This can be your own image pair that you believe
    will give you the best stitched panorama. Feel free to play around with 
    different image pairs as a fun exercise!
    
    Please note that you can use your fundamental matrix estimation from part3
    (imported for you above) to compute the homography matrix that you will 
    need to stitch the panorama.
    
    Feel free to reuse your interest point pipeline from project 2, or you may
    choose to use any existing interest point/feature matching functions from
    OpenCV. You may NOT use any pre-existing warping function though.

    Args:
        imageA: first image that we are looking at (from camera view 1) [A x B]
        imageB: second image that we are looking at (from camera view 2) [M x N]

    Returns:
        panorama: stitch of image 1 and image 2 using warp. Ideal dimensions
            are either:
            1. A or M x (B + N)
                    OR
            2. (A + M) x B or N)
    """
    panorama = None

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    imageA=cv.imread(imageA)

    imageB = cv.imread(imageB)

    imageA_g=cv.cvtColor(imageA,cv.COLOR_BGR2GRAY)
    imageB_g = cv.cvtColor(imageB, cv.COLOR_BGR2GRAY)

    
    my_sift = cv.xfeatures2d.SIFT_create()
    kp1, des1 = my_sift.detectAndCompute(imageA_g, None)
    kp2, des2 = my_sift.detectAndCompute(imageB_g, None)

    bf = cv.BFMatcher()

    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m in matches:
        if (m[0].distance < 0.5 * m[1].distance):
            good.append(m)
    matches = np.asarray(good)

    #matches=ransac_fundamental_matrix(imageA,imageB)

    dst = np.float32([kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
    src = np.float32([kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)

    H, masked = cv.findHomography(src, dst, cv.RANSAC, 5.0)

    def to_tx(img):
        H, V, C = img.shape
        mtr = np.zeros((V, H, C), dtype='int')
        for i in range(img.shape[0]):
            mtr[:, i] = img[i]

        return mtr

    def to_img(mtr):
        V, H, C = mtr.shape
        img = np.zeros((H, V, C), dtype='int')
        for i in range(mtr.shape[0]):
            img[:, i] = mtr[i]

        return img
    def warpPerspective(img, M, dsize):
        mtr = to_tx(img)
        R, C = dsize
        dst = np.zeros((R, C, mtr.shape[2]))
        for i in range(mtr.shape[0]):
            for j in range(mtr.shape[1]):
                res = np.dot(M, [i, j, 1])
                i2, j2, _ = (res / res[2] + 0.5).astype(int)
                if i2 >= 0 and i2 < R:
                    if j2 >= 0 and j2 < C:
                        dst[i2, j2] = mtr[i, j]

        return to_img(dst)

    dst = warpPerspective(imageA, H, ((imageA.shape[1] + imageB.shape[1]), imageB.shape[0]))  
    dst[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
    panorama=dst
    cv.imwrite('additional_data/output.jpg', dst)
    #plt.imshow(dst)
    #plt.show()

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return panorama
