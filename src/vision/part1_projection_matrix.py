import numpy as np


def calculate_projection_matrix(
    points_2d: np.ndarray, points_3d: np.ndarray
) -> np.ndarray:
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points:

                                                      [ M11      [ u1
                                                        M12        v1
                                                        M13        .
                                                        M14        .
    [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1        M21        .
      0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1        M22        .
      .  .  .  . .  .  .  .    .     .      .       *   M23   =    .
      Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn        M24        .
      0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]      M31        .
                                                        M32        un
                                                        M33 ]      vn ]

    Then you can solve this using least squares with np.linalg.lstsq() or SVD.
    Notice you obtain 2 equations for each corresponding 2D and 3D point
    pair. To solve this, you need at least 6 point pairs.

    Args:
        points_2d: A numpy array of shape (N, 2)
        points_3d: A numpy array of shape (N, 3)

    Returns:
        M: A numpy array of shape (3, 4) representing the projection matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    N=points_3d.shape[0]
    B=np.ones((points_3d.shape[0],points_3d.shape[1]+1))
    B[:points_3d.shape[0],:points_3d.shape[1]]=points_3d
    A=np.zeros((2*N,12))
    for i in range(N):
        A[2*i]=[B[i,0],B[i,1],B[i,2],B[i,3],0,0,0,0,-B[i,0]*points_2d[i,0],-B[i,1]*points_2d[i,0],-B[i,2]*points_2d[i,0],-B[i,3]*points_2d[i,0]]
        A[2 * i+1] = [0, 0, 0, 0,B[i, 0], B[i, 1], B[i, 2], B[i, 3],  -B[i, 0] * points_2d[i, 1],-B[i, 1] * points_2d[i, 1], -B[i, 2] * points_2d[i, 1], -B[i, 3] * points_2d[i, 1]]
    Y=np.zeros(2*N)
    U,S,V=np.linalg.svd(A)
    M2=V[-1]
    M=M2.reshape((3,4))



    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return M


def projection(P: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
    Computes projection from [X,Y,Z] in non-homogenous coordinates to
    (x,y) in non-homogenous image coordinates.
    Args:
        P: 3 x 4 projection matrix
        points_3d: n x 3 array of points [X_i,Y_i,Z_i]
    Returns:
        projected_points_2d: n x 2 array of points in non-homogenous image
            coordinates
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    B = np.ones((points_3d.shape[0], points_3d.shape[1] + 1))
    B[:points_3d.shape[0], :points_3d.shape[1]] = points_3d
    B=B.T
    S=P.dot(B)
    S[0]=S[0]/S[2]
    S[1]=S[1]/S[2]
    projected_points_2d=np.zeros((points_3d.shape[0],2))
    projected_points_2d[:,0]=S[0]
    projected_points_2d[:, 1] = S[1]


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return projected_points_2d


def calculate_camera_center(M: np.ndarray) -> np.ndarray:
    """
    Returns the camera center matrix for a given projection matrix.

    Args:
    -   M: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    Q=M[:,:3]
    m4=M[:,3]
    cc=-np.linalg.inv(Q).dot(m4)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return cc
