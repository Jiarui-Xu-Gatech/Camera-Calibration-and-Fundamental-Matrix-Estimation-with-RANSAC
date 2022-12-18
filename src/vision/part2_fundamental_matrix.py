"""Fundamental matrix utilities."""

import numpy as np


def normalize_points(points: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Perform coordinate normalization through linear transformations.
    Args:
        points: A numpy array of shape (N, 2) representing the 2D points in
            the image

    Returns:
        points_normalized: A numpy array of shape (N, 2) representing the
            normalized 2D points in the image
        T: transformation matrix representing the product of the scale and
            offset matrices
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    cu, cv = np.mean(points, axis=0)
    U=points[:,0]-np.ones(points.shape[0])*cu
    V=points[:,1]-np.ones(points.shape[0])*cv
    su = 1.0 / np.std(U)
    sv = 1.0 / np.std(V)

    scale_matrix=np.zeros((3,3))
    offset_matrix=np.zeros((3,3))
    scale_matrix[0,0]=su
    scale_matrix[1,1]=sv
    scale_matrix[2,2]=1
    offset_matrix[0,0]=1
    offset_matrix[0,2]=-cu
    offset_matrix[1,1]=1
    offset_matrix[1,2]=-cv
    offset_matrix[2,2]=1
    T = scale_matrix.dot(offset_matrix)

    points_1s = np.ones([len(points), 3])
    points_1s[:, 0:2] = points

    points_normalized_1s = points_1s.dot(T.T)  # (N,3).dot(3,3) = (N,3)
    points_normalized = points_normalized_1s[:, 0:2]



    '''
    U,S,V=np.linalg.svd(points)
    print(S.shape)
    print(U.shape)
    print(V.shape)
    points_normalized=S
    T=U.T
    '''

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return points_normalized, T


def unnormalize_F(F_norm: np.ndarray, T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """
    Adjusts F to account for normalized coordinates by using the transformation
    matrices.

    Args:
        F_norm: A numpy array of shape (3, 3) representing the normalized
            fundamental matrix
        T_a: Transformation matrix for image A
        T_B: Transformation matrix for image B

    Returns:
        F_orig: A numpy array of shape (3, 3) representing the original
            fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    F_orig=((T_b.T).dot(F_norm)).dot(T_a)

    ###########################################################################
    #                             END OF YOUR CODE
    ###########################################################################

    return F_orig


def estimate_fundamental_matrix(
    points_a: np.ndarray, points_b: np.ndarray
) -> np.ndarray:
    """
    Calculates the fundamental matrix. You may use the normalize_points() and
    unnormalize_F() functions here.

    Args:
        points_a: A numpy array of shape (N, 2) representing the 2D points in
            image A
        points_b: A numpy array of shape (N, 2) representing the 2D points in
            image B

    Returns:
        F: A numpy array of shape (3, 3) representing the fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    '''
    N=points_a.shape[0]
    A=np.ones((N,9))
    for i in range(N):
        A[i]=[points_a[i,0]*points_b[i,0],points_a[i,0]*points_b[i,1],points_a[i,0],points_a[i,1]*points_a[i,0],points_a[i,1]*points_b[i,1],points_a[i,1],points_b[i,0],points_b[i,1],1]
    U,S,V=np.linalg.svd(A)
    f = V[-1]
    F = f.reshape((3, 3))
    '''

    A = np.zeros([len(points_a), 8])
    norm_a, T_a = normalize_points(points_a)
    norm_b, T_b = normalize_points(points_b)

    for i in range(norm_a.shape[0]):
        A[i] = [norm_a[i, 0] * norm_b[i, 0], norm_a[i, 1] * norm_b[i, 0], norm_b[i, 0],
                norm_a[i, 0] * norm_b[i, 1], norm_a[i, 1] * norm_b[i, 1], norm_b[i, 1], norm_a[i, 0],
                norm_a[i, 1]]
    Y = -np.ones(A.shape[0])
    F, _0,_1,_2 = np.linalg.lstsq(A, Y, rcond=None)
    F = np.append(F, 1)
    F=F.reshape((3,3))

    U, S, Vh = np.linalg.svd(F)
    S = np.diag(S)
    S[2, 2] = 0

    F_norm = (U.dot(S)).dot(Vh)

    F = unnormalize_F(F_norm, T_a, T_b)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F
