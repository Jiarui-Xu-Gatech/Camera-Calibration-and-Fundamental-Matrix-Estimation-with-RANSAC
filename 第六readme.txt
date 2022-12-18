1.I loaded the images and converted the images to grayscale.

2.Then I computed the SIFT keypoints and descriptors
A Scale Invariant Feature Transform (SIFT) feature or keypoint, is a selected circular region of an image with an orientation. 

3.Find Top M matches of descriptors of 2 images
Now that I have SIFT keypoints and descriptors, I need to find the distance of each descriptor of image 1 to each descriptor of image 2.
bf = cv.BFMatcher()

4.I select the top M matches of the descriptors. Here, I took the value of M to be 2.
matches = bf.knnMatch(des1,des2, k=2)
Even in the top 2 descriptors, I may have obtained some trivial descriptors. I eliminate those with ratio test.

5.Choose/get interest points for the pair of images. 
I aligned the 2 images using homography transformation

6.Find candidate matches among the interest points. Also use RANSAC.
I calculated the homography matrix of the 2 images, which require atleast 4 matches, to align the images. It uses Random Sample Consesus (RANSAC), which produces right results even in presence of bad matches.

7.function to_tx() and to_img(mtr) are for changing the representation in order to make my implementation more simple

8.Implementing warpPerspective. As for affine transform, some pixels from original image might be mapped outside or into subwindow of the original image.

9.Project each image onto the same surface and stitch (warp operation)
Once I have the homography for transformation, I can warp the image and stitch the two images.

