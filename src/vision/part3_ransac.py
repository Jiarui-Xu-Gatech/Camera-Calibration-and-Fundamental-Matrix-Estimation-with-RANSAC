import math

import numpy as np
from vision.part2_fundamental_matrix import estimate_fundamental_matrix


def calculate_num_ransac_iterations(
    prob_success: float, sample_size: int, ind_prob_correct: float
) -> int:
    """
    Calculates the number of RANSAC iterations needed for a given guarantee of
    success.

    Args:
        prob_success: float representing the desired guarantee of success
        sample_size: int the number of samples included in each RANSAC
            iteration
        ind_prob_success: float representing the probability that each element
            in a sample is correct

    Returns:
        num_samples: int the number of RANSAC iterations needed

    """
    num_samples = None
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    num_samples=np.log(1-prob_success)/np.log(1-ind_prob_correct**sample_size)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return int(num_samples)


def ransac_fundamental_matrix(
    matches_a: np.ndarray, matches_b: np.ndarray
) -> np.ndarray:
    """
    For this section, use RANSAC to find the best fundamental matrix by
    randomly sampling interest points. You would reuse
    estimate_fundamental_matrix() from part 2 of this assignment and
    calculate_num_ransac_iterations().

    If you are trying to produce an uncluttered visualization of epipolar
    lines, you may want to return no more than 30 points for either left or
    right images.

    Tips:
        0. You will need to determine your prob_success, sample_size, and
            ind_prob_success values. What is an acceptable rate of success? How
            many points do you want to sample? What is your estimate of the
            correspondence accuracy in your dataset?
        1. A potentially useful function is numpy.random.choice for creating
            your random samples.
        2. You will also need to choose an error threshold to separate your
            inliers from your outliers. We suggest a threshold of 0.1.

    Args:
        matches_a: A numpy array of shape (N, 2) representing the coordinates
            of possibly matching points from image A
        matches_b: A numpy array of shape (N, 2) representing the coordinates
            of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)

    Returns:
        best_F: A numpy array of shape (3, 3) representing the best fundamental
            matrix estimation
        inliers_a: A numpy array of shape (M, 2) representing the subset of
            corresponding points from image A that are inliers with respect to
            best_F
        inliers_b: A numpy array of shape (M, 2) representing the subset of
            corresponding points from image B that are inliers with respect to
            best_F
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    inliers_a = []
    inliers_b = []


    prob_success = 0.999
    sample_size = 8
    ind_prob_correct = 0.9
    threshold = 0.1
    num_iterations = calculate_num_ransac_iterations(prob_success, sample_size, ind_prob_correct)
    N = matches_a.shape[0]
    M = 0

    for i in range(num_iterations):
        num = 0
        sample_inliers_a = []
        sample_inliers_b = []
        idx = np.random.choice(N, sample_size, replace=False)
        sample_points_b = matches_b[idx, :]
        sample_points_a = matches_a[idx, :]
        F = estimate_fundamental_matrix(sample_points_a, sample_points_b)
        for j in range(N):
            dist = np.linalg.norm(np.append(matches_a[j], 1).dot(F).dot(np.append(matches_b[j], 1).T))
            if dist < threshold:
                num += 1
                sample_inliers_b.append(matches_b[j])
                sample_inliers_a.append(matches_a[j])

        if num > M:
            best_F = F
            inliers_a = sample_inliers_a
            inliers_b = sample_inliers_b
            M = num



        inliers_a = np.array(inliers_a)
        inliers_b = np.array(inliers_b)
    

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return best_F, inliers_a, inliers_b
